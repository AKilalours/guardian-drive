"""
carla_agent/reward/reward_fn.py
Guardian Drive Reward Function

Translates safety events into scalar rewards for RL training.
Designed to train an agent that:
  - Does not over-escalate (cry-wolf penalty)
  - Does not under-escalate (missed alarm penalty)
  - Drives smoothly (jerk + lane penalty)
  - Correctly routes to rest stops at appropriate severity

This is the safety-critical reward design — changing it changes
the entire learned policy.

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


# ─────────────────────────────────────────────
# Impairment classifier (mirrors Guardian Drive core)
# ─────────────────────────────────────────────

def classify_impairment(
    ear: float,
    perclos: float,
    hrv_rmssd: float,
    ecg_hr: int,
    spo2: float,
    yawn_count: int,
    drive_seconds: int,
    g_peak: float,
) -> str:
    """Deterministic impairment classifier — ground truth for reward."""
    # Crash
    if g_peak >= 2.0:
        return "CRASH"
    # Microsleep (override)
    if ear < 0.15 or perclos > 0.80:
        return "MICROSLEEP"
    # Stroke suspect
    if spo2 < 92 and hrv_rmssd < 15:
        return "STROKE_SUSPECT"
    # Cardiac
    if ecg_hr > 120 or ecg_hr < 45:
        return "CARDIAC_ALERT"
    # Sleepy
    if perclos > 0.25 and yawn_count >= 3:
        return "SLEEPY"
    # Fatigued
    if hrv_rmssd < 20 or drive_seconds > 90 * 60:
        return "FATIGUED"
    # Drowsy
    if perclos > 0.15:
        return "DROWSY"
    return "ALERT"


# Ground truth: what alert level each impairment SHOULD trigger
CORRECT_RESPONSE = {
    "ALERT":          0,   # NOMINAL
    "DROWSY":         1,   # ADVISORY
    "SLEEPY":         2,   # CAUTION
    "FATIGUED":       1,   # ADVISORY
    "CARDIAC_ALERT":  3,   # PULLOVER
    "STROKE_SUSPECT": 4,   # ESCALATE
    "MICROSLEEP":     4,   # ESCALATE
    "CRASH":          4,   # ESCALATE
}

ACTION_LABELS = ["NOMINAL", "ADVISORY", "CAUTION", "PULLOVER", "ESCALATE"]


# ─────────────────────────────────────────────
# Reward components
# ─────────────────────────────────────────────

@dataclass
class RewardConfig:
    """Tunable reward weights."""
    # Safety accuracy
    correct_response: float = +3.0      # agent chose right alert level
    under_escalate: float = -5.0        # agent too calm when emergency
    over_escalate: float = -1.5         # agent too aggressive when safe
    # Driving quality
    collision_penalty: float = -20.0
    lane_invasion_penalty: float = -3.0
    jerk_penalty_scale: float = -0.05   # per m/s³
    speed_bonus_scale: float = +0.01    # per km/h (encourages progress)
    # Fault resilience
    ecg_dropout_correct: float = +1.0   # SQI abstain under dropout
    gps_loss_handled: float = +0.5      # graceful degradation
    # Survival bonus
    alive_bonus: float = +0.1           # per step


class GuardianReward:
    """
    Computes scalar reward from SensorBundle + agent action.

    Used for both PPO training and evaluation metric.
    """

    def __init__(self, config: RewardConfig = None):
        self.cfg = config or RewardConfig()

    def compute(
        self,
        bundle: Any,  # SensorBundle
        action: int,
        stats: Any,   # EpisodeStats
    ) -> Tuple[float, Dict[str, float]]:
        """Return (reward, reward_breakdown_dict)."""

        # Ground truth impairment
        gt_impairment = classify_impairment(
            ear=bundle.ear,
            perclos=bundle.perclos,
            hrv_rmssd=bundle.hrv_rmssd,
            ecg_hr=bundle.ecg_hr,
            spo2=bundle.spo2,
            yawn_count=bundle.yawn_count,
            drive_seconds=getattr(stats, "_drive_seconds", 0) if hasattr(stats, "_drive_seconds") else 0,
            g_peak=bundle.g_peak,
        )
        correct_action = CORRECT_RESPONSE[gt_impairment]
        components: Dict[str, float] = {}

        # ── Safety accuracy ──────────────────────────────────
        delta = action - correct_action
        if delta == 0:
            components["safety_accuracy"] = self.cfg.correct_response
        elif delta < 0:
            # Under-escalation: agent too calm
            severity = correct_action  # 0-4
            components["safety_accuracy"] = self.cfg.under_escalate * (abs(delta) * severity / 4.0 + 1.0)
        else:
            # Over-escalation: unnecessary alert
            components["safety_accuracy"] = self.cfg.over_escalate * abs(delta)

        # ── Driving quality ───────────────────────────────────
        components["collision"] = (
            self.cfg.collision_penalty if bundle.collision_intensity > 0 else 0.0
        )
        components["lane_invasion"] = (
            self.cfg.lane_invasion_penalty if bundle.lane_invaded else 0.0
        )
        components["jerk"] = self.cfg.jerk_penalty_scale * bundle.jerk_peak
        components["speed"] = self.cfg.speed_bonus_scale * bundle.speed_kph

        # ── Fault resilience ──────────────────────────────────
        if bundle.ecg_dropout:
            # Under ECG dropout, agent should NOT escalate purely on ECG
            # Correct: use other sensors, don't cry wolf
            if action <= 1:  # NOMINAL or ADVISORY — appropriate caution
                components["fault_resilience"] = self.cfg.ecg_dropout_correct
            else:
                components["fault_resilience"] = -0.5  # over-reaction to dropout

        if bundle.gps_loss:
            # Under GPS loss, agent should still respond to physiology
            components["gps_handling"] = self.cfg.gps_loss_handled

        # ── Survival bonus ────────────────────────────────────
        if bundle.collision_intensity == 0:
            components["alive"] = self.cfg.alive_bonus

        total = sum(components.values())

        return total, {
            **components,
            "gt_impairment": gt_impairment,
            "correct_action": correct_action,
            "chosen_action": action,
            "action_label": ACTION_LABELS[action],
            "total": total,
        }
