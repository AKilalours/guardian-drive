"""
policy/rl_agent.py
Guardian Drive -- Safety Policy as Reinforcement Learning Agent

The Guardian Drive safety state machine IS a policy:
  - State:  (r_total, SQI, impairment_type, AV_context, drive_minutes)
  - Action: (state_level, POI_type, vibration, voice, emergency_dispatch)
  - Reward: negative if escalation happens, positive if driver improves

This module documents the RL framing and implements a simple
Q-table policy that can be trained offline on telemetry logs.

Why this matters for Tesla:
- FSD safety supervisor uses RL to learn intervention policies
- Offline RL from logged data matches Tesla's data-driven approach
- The ESCALATE/PULLOVER/ADVISORY hierarchy IS a reward-shaping structure

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

# ── State / Action spaces ─────────────────────────────────────────
STATES  = ["NOMINAL","ADVISORY","CAUTION","PULLOVER","ESCALATE"]
ACTIONS = ["monitor","suggest_rest","strong_alert","pullover_now","emergency"]

# Reward structure -- negative reward for escalation, positive for recovery
REWARDS = {
    ("NOMINAL",   "monitor"):       +1.0,
    ("ADVISORY",  "suggest_rest"):  +0.5,
    ("CAUTION",   "strong_alert"):  +0.5,
    ("PULLOVER",  "pullover_now"):  +0.3,
    ("ESCALATE",  "emergency"):     -1.0,  # reached emergency -- negative
    # Wrong actions
    ("ADVISORY",  "monitor"):       -0.5,  # missed advisory
    ("ESCALATE",  "monitor"):       -5.0,  # very bad -- ignored emergency
}

class SafetyPolicyAgent:
    """
    Tabular Q-learning agent for safety intervention policy.
    Trained offline on TelemetryLogger JSONL logs.
    
    This is offline RL / behavioral cloning from expert demonstrations
    (the hand-crafted state machine = expert policy).
    """
    
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.alpha   = alpha    # learning rate
        self.gamma   = gamma    # discount factor
        self.epsilon = epsilon  # exploration
        # Q-table: states x actions
        self.Q = np.zeros((len(STATES), len(ACTIONS)))
        self._state_idx = {s:i for i,s in enumerate(STATES)}
        self._action_idx = {a:i for i,a in enumerate(ACTIONS)}

    def discretize_state(self, r_total: float, sqi: float,
                          impairment: str, drive_mins: float) -> str:
        """Map continuous features to discrete state."""
        if r_total > 0.75 or impairment == "microsleep":
            return "ESCALATE"
        elif r_total > 0.55 or impairment == "sleepy":
            return "PULLOVER"
        elif r_total > 0.40 or impairment == "drowsy":
            return "CAUTION"
        elif r_total > 0.25 or impairment == "fatigued":
            return "ADVISORY"
        return "NOMINAL"

    def select_action(self, state: str) -> str:
        """Epsilon-greedy action selection."""
        si = self._state_idx[state]
        if np.random.random() < self.epsilon:
            return ACTIONS[np.random.randint(len(ACTIONS))]
        return ACTIONS[np.argmax(self.Q[si])]

    def update(self, state: str, action: str,
               reward: float, next_state: str):
        """Q-learning update."""
        si  = self._state_idx[state]
        ai  = self._action_idx[action]
        nsi = self._state_idx[next_state]
        td  = reward + self.gamma*np.max(self.Q[nsi]) - self.Q[si,ai]
        self.Q[si,ai] += self.alpha * td

    def train_from_logs(self, log_path: str, episodes: int = 100):
        """Train from TelemetryLogger JSONL replay."""
        import json
        from pathlib import Path
        if not Path(log_path).exists():
            print(f"No log found at {log_path} -- using synthetic demo")
            self._demo_train()
            return
        frames = [json.loads(l) for l in
                  open(log_path) if l.strip()]
        print(f"Training on {len(frames)} telemetry frames")
        for ep in range(episodes):
            total_reward = 0
            for i in range(len(frames)-1):
                f  = frames[i]
                fn = frames[i+1]
                state      = f.get("level","NOMINAL")
                next_state = fn.get("level","NOMINAL")
                action     = ACTIONS[min(
                    self._state_idx.get(state,0),
                    len(ACTIONS)-1)]
                reward = REWARDS.get((state,action),0)
                self.update(state, action, reward, next_state)
                total_reward += reward
            if (ep+1) % 10 == 0:
                print(f"  Episode {ep+1}/{episodes} "
                      f"total_reward={total_reward:.1f}")

    def _demo_train(self):
        """Demo training with synthetic transitions."""
        transitions = [
            ("NOMINAL","monitor",+1.0,"NOMINAL"),
            ("ADVISORY","suggest_rest",+0.5,"NOMINAL"),
            ("CAUTION","strong_alert",+0.5,"ADVISORY"),
            ("PULLOVER","pullover_now",+0.3,"CAUTION"),
            ("ESCALATE","emergency",-1.0,"PULLOVER"),
        ]*100
        for s,a,r,ns in transitions:
            self.update(s,a,r,ns)
        print("Demo Q-table trained")

    def print_policy(self):
        print("\nLearned Safety Policy (Q-table):")
        print(f"{'State':12} {'Best Action':20} Q-value")
        for s in STATES:
            si  = self._state_idx[s]
            ai  = np.argmax(self.Q[si])
            print(f"  {s:12} {ACTIONS[ai]:20} {self.Q[si,ai]:.3f}")


if __name__ == "__main__":
    agent = SafetyPolicyAgent()
    agent.train_from_logs(
        "data/telemetry_logs/session_latest.jsonl")
    agent.print_policy()
