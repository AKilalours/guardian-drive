"""
carla_agent/agent/policy.py
Guardian Drive Safety Policy

Two-stage training:
  Stage 1: Behavior Cloning (BC) from expert demonstrations
  Stage 2: PPO fine-tuning with Guardian Drive reward

Architecture: MLP policy network
  Input:  20-dim observation (9 sensors + derived features)
  Output: 5-class action (NOMINAL / ADVISORY / CAUTION / PULLOVER / ESCALATE)

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Policy network
# ─────────────────────────────────────────────

class GuardianPolicyNet(nn.Module):
    """
    Safety policy network.
    Maps 20-dim sensor observation → 5-class action distribution.

    Architecture:
        FC(20→128) → LayerNorm → ReLU
        FC(128→128) → LayerNorm → ReLU
        FC(128→64)  → ReLU
        FC(64→5)    → logits

    Also outputs a value head for PPO:
        FC(64→1) → scalar value estimate
    """

    def __init__(self, obs_dim: int = 20, n_actions: int = 5):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(64, n_actions)
        self.value_head = nn.Linear(64, 1)

        # Initialise last layer with small weights for stable start
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (B, 5) unnormalised action logits
            values: (B,)   state value estimate
        """
        features = self.shared(obs)
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return logits, values

    def get_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        Sample or argmax action from observation.
        Returns (action, log_prob, value).
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.forward(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1).item()
        else:
            action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action)).item()
        return int(action), log_prob, value.item()


# ─────────────────────────────────────────────
# Expert policy (rule-based, for BC demos)
# ─────────────────────────────────────────────

class ExpertPolicy:
    """
    Rule-based expert policy for generating BC demonstrations.
    Maps observation vector → correct action using Guardian Drive rules.
    This is the supervisor in DAgger/BC training.
    """

    def __call__(self, obs: np.ndarray) -> int:
        """
        obs layout (matches carla_env.py _bundle_to_obs):
            0: ear
            1: perclos
            2: yawn_count
            3: facial_asymmetry
            4: hrv_rmssd / 80
            5: ecg_hr / 160
            6: spo2 / 100
            7: gsr_us / 20
            8: g_peak / 6
            9: jerk_peak / 30
            10: steering_delta / 90
            11: cabin_temp / 40
            12: speech_clarity
            13: speed / 130
            14: collision flag
            15: lane_invasion flag
            16: ecg_dropout flag
            17: gps_loss flag
            18: camera_occluded flag
            19: drive_time / (90*60)
        """
        ear = obs[0]
        perclos = obs[1]
        yawns = int(obs[2] * 10)  # denormalise approximately
        hrv = obs[4] * 80.0
        ecg_hr = int(obs[5] * 160)
        spo2 = obs[6] * 100.0
        g_peak = obs[8] * 6.0
        collision = obs[14] > 0.5
        drive_frac = obs[19]

        # Crash override
        if collision or g_peak >= 2.0:
            return 4  # ESCALATE

        # Stroke / hypoxia
        if spo2 < 92.0:
            return 4  # ESCALATE

        # Microsleep
        if ear < 0.15 or perclos > 0.80:
            return 4  # ESCALATE

        # Cardiac
        if ecg_hr > 120 or ecg_hr < 45:
            return 3  # PULLOVER

        # Sleepy
        if perclos > 0.25 and yawns >= 3:
            return 2  # CAUTION

        # Fatigued
        if hrv < 20 or drive_frac > 1.0:
            return 1  # ADVISORY

        # Drowsy
        if perclos > 0.15:
            return 1  # ADVISORY

        return 0  # NOMINAL


# ─────────────────────────────────────────────
# Behavior Cloning trainer
# ─────────────────────────────────────────────

class BehaviorCloningTrainer:
    """
    Stage 1: Supervised imitation from ExpertPolicy demonstrations.

    Rollout expert → collect (obs, expert_action) pairs → train cross-entropy.
    """

    def __init__(
        self,
        policy: GuardianPolicyNet,
        lr: float = 3e-4,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.expert = ExpertPolicy()
        self.losses: list = []

    def collect_demos(self, env, n_steps: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        """Roll out expert policy to collect BC dataset."""
        obs_list, action_list = [], []
        obs, _ = env.reset()
        for _ in range(n_steps):
            action = self.expert(obs)
            obs_list.append(obs.copy())
            action_list.append(action)
            obs, _, done, truncated, _ = env.step(action)
            if done or truncated:
                obs, _ = env.reset()
        return np.array(obs_list, dtype=np.float32), np.array(action_list, dtype=np.int64)

    def train(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        n_epochs: int = 20,
        batch_size: int = 256,
    ) -> Dict[str, list]:
        """Train policy on expert demonstrations."""
        n = len(obs)
        obs_t = torch.FloatTensor(obs).to(self.device)
        act_t = torch.LongTensor(actions).to(self.device)

        metrics = {"loss": [], "accuracy": []}
        for epoch in range(n_epochs):
            idx = torch.randperm(n)
            epoch_losses, epoch_accs = [], []
            for start in range(0, n, batch_size):
                batch_idx = idx[start:start + batch_size]
                batch_obs = obs_t[batch_idx]
                batch_act = act_t[batch_idx]

                self.optimizer.zero_grad()
                logits, _ = self.policy(batch_obs)
                loss = F.cross_entropy(logits, batch_act)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    acc = (preds == batch_act).float().mean().item()

                epoch_losses.append(loss.item())
                epoch_accs.append(acc)

            metrics["loss"].append(float(np.mean(epoch_losses)))
            metrics["accuracy"].append(float(np.mean(epoch_accs)))

            if (epoch + 1) % 5 == 0:
                print(f"  BC Epoch {epoch+1}/{n_epochs}: "
                      f"loss={metrics['loss'][-1]:.4f}, "
                      f"acc={metrics['accuracy'][-1]:.3f}")

        return metrics

    def save(self, path: str) -> None:
        torch.save({
            "model_state": self.policy.state_dict(),
            "stage": "behavior_cloning",
        }, path)
        print(f"[BC] Saved to {path}")


# ─────────────────────────────────────────────
# PPO Trainer (Stage 2)
# ─────────────────────────────────────────────

class PPOTrainer:
    """
    Stage 2: PPO fine-tuning of BC-initialised policy.
    Uses Guardian Drive reward for closed-loop safety optimisation.

    References:
    - Schulman et al., Proximal Policy Optimization Algorithms (2017)
    """

    def __init__(
        self,
        policy: GuardianPolicyNet,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        n_steps: int = 512,
        n_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
        self.train_metrics: list = []

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generalised Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = 0.0 if dones[t] else values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values[:len(rewards)]
        return advantages, returns

    def collect_rollout(self, env) -> Dict[str, np.ndarray]:
        """Collect n_steps of experience."""
        obs_buf = np.zeros((self.n_steps, self.policy.obs_dim), dtype=np.float32)
        act_buf = np.zeros(self.n_steps, dtype=np.int64)
        logp_buf = np.zeros(self.n_steps, dtype=np.float32)
        rew_buf = np.zeros(self.n_steps, dtype=np.float32)
        val_buf = np.zeros(self.n_steps + 1, dtype=np.float32)
        done_buf = np.zeros(self.n_steps, dtype=np.float32)

        obs, _ = env.reset()
        for t in range(self.n_steps):
            action, logp, value = self.policy.get_action(obs)
            obs_buf[t] = obs
            act_buf[t] = action
            logp_buf[t] = logp
            val_buf[t] = value

            obs, reward, done, truncated, _ = env.step(action)
            rew_buf[t] = reward
            done_buf[t] = float(done or truncated)

            if done or truncated:
                obs, _ = env.reset()

        _, _, val_buf[-1] = self.policy.get_action(obs)
        advantages, returns = self.compute_gae(rew_buf, val_buf, done_buf)
        return {
            "obs": obs_buf, "actions": act_buf, "log_probs": logp_buf,
            "advantages": advantages, "returns": returns,
        }

    def update(self, rollout: Dict[str, np.ndarray]) -> Dict[str, float]:
        """PPO update step."""
        obs_t = torch.FloatTensor(rollout["obs"]).to(self.device)
        act_t = torch.LongTensor(rollout["actions"]).to(self.device)
        old_logp_t = torch.FloatTensor(rollout["log_probs"]).to(self.device)
        adv_t = torch.FloatTensor(rollout["advantages"]).to(self.device)
        ret_t = torch.FloatTensor(rollout["returns"]).to(self.device)

        # Normalise advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = len(obs_t)
        metrics = {"policy_loss": [], "value_loss": [], "entropy": []}

        for _ in range(self.n_epochs):
            idx = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                b = idx[start:start + self.batch_size]
                logits, values = self.policy(obs_t[b])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act_t[b])
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - old_logp_t[b])
                clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                policy_loss = -torch.min(ratio * adv_t[b], clipped * adv_t[b]).mean()
                value_loss = F.mse_loss(values, ret_t[b])
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.item())

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def train(self, env, n_updates: int = 100) -> list:
        """Full PPO training loop."""
        all_metrics = []
        for update in range(n_updates):
            rollout = self.collect_rollout(env)
            metrics = self.update(rollout)
            metrics["update"] = update
            all_metrics.append(metrics)
            if (update + 1) % 10 == 0:
                print(f"  PPO Update {update+1}/{n_updates}: "
                      f"policy_loss={metrics['policy_loss']:.4f}, "
                      f"value_loss={metrics['value_loss']:.4f}, "
                      f"entropy={metrics['entropy']:.4f}")
        return all_metrics

    def save(self, path: str) -> None:
        torch.save({
            "model_state": self.policy.state_dict(),
            "stage": "ppo_finetuned",
        }, path)
        print(f"[PPO] Saved to {path}")
