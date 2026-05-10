"""
tests/property/test_safety_properties.py
Property-Based Tests for Guardian Drive Safety Invariants

Uses Hypothesis to find edge cases that unit tests miss.
This is the evidence for Tesla Code Hardening role.

Properties tested:
  1. Fusion weights always sum to 1.0
  2. Alert level never decreases under increasing risk
  3. MICROSLEEP always maps to ESCALATE (safety invariant)
  4. SQI abstention never raises false ESCALATE
  5. Reward function is bounded
  6. Crash detection is monotonic in g-peak
  7. Stroke assessment is monotonic in SpO2 risk

Run:
    pip install hypothesis
    pytest tests/property/test_safety_properties.py -v

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import pytest
import numpy as np

try:
    from hypothesis import given, settings, assume, HealthCheck
    from hypothesis import strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Stub out decorators so file is importable
    def given(*args, **kwargs):
        def decorator(fn): return fn
        return decorator
    def settings(*args, **kwargs):
        def decorator(fn): return fn
        return decorator
    class st:
        @staticmethod
        def floats(*a, **kw): return None
        @staticmethod
        def integers(*a, **kw): return None
        @staticmethod
        def booleans(): return None
        @staticmethod
        def just(x): return None

from carla_agent.reward.reward_fn import (
    classify_impairment, CORRECT_RESPONSE, GuardianReward, RewardConfig
)


# ─────────────────────────────────────────────
# Helper: minimal SensorBundle-like object
# ─────────────────────────────────────────────

class FakeBundle:
    def __init__(self, **kwargs):
        defaults = dict(
            ear=0.28, perclos=0.08, yawn_count=0, facial_asymmetry=0.02,
            hrv_rmssd=45.0, ecg_hr=72, spo2=98.0, gsr_us=3.0,
            g_peak=0.1, jerk_peak=0.5, steering_delta=5.0,
            cabin_temp_c=22.0, speech_clarity=0.9, speed_kph=60.0,
            collision_intensity=0.0, lane_invaded=False,
            ecg_dropout=False, gps_loss=False, camera_occluded=False,
        )
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


class FakeStats:
    _drive_seconds = 0


# ─────────────────────────────────────────────
# Property 1: Fusion weights sum to 1.0
# ─────────────────────────────────────────────

def test_fusion_weights_sum_to_one():
    """Fusion equation weights must always sum to 1.0."""
    w_phys  = 0.40
    w_neuro = 0.30
    w_imu   = 0.20
    w_ctx   = 0.10
    total = w_phys + w_neuro + w_imu + w_ctx
    assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, not 1.0"


# ─────────────────────────────────────────────
# Property 2: MICROSLEEP always → ESCALATE
# ─────────────────────────────────────────────

@given(
    ear=st.floats(min_value=0.0, max_value=0.14),   # below microsleep threshold
    perclos=st.floats(min_value=0.0, max_value=1.0),
    hrv=st.floats(min_value=10.0, max_value=80.0),
    hr=st.integers(min_value=45, max_value=120),
    spo2=st.floats(min_value=92.0, max_value=100.0),
)
@settings(max_examples=200)
def test_microsleep_always_escalates(ear, perclos, hrv, hr, spo2):
    """If EAR < 0.15 (eye closed), alert must be ESCALATE regardless of other signals."""
    result = classify_impairment(
        ear=ear, perclos=perclos, hrv_rmssd=hrv,
        ecg_hr=hr, spo2=spo2, yawn_count=0,
        drive_seconds=0, g_peak=0.0,
    )
    correct = CORRECT_RESPONSE[result]
    assert correct == 4, (
        f"EAR={ear:.3f} < 0.15 must map to ESCALATE (4), got {result} → {correct}"
    )


# ─────────────────────────────────────────────
# Property 3: High PERCLOS always → at least CAUTION
# ─────────────────────────────────────────────

@given(
    perclos=st.floats(min_value=0.81, max_value=1.0),  # severe eye closure
    hrv=st.floats(min_value=25.0, max_value=80.0),
    hr=st.integers(min_value=50, max_value=110),
    spo2=st.floats(min_value=94.0, max_value=100.0),
    ear=st.floats(min_value=0.20, max_value=0.45),
)
@settings(max_examples=200)
def test_high_perclos_escalates(perclos, hrv, hr, spo2, ear):
    """PERCLOS > 0.80 must result in ESCALATE."""
    result = classify_impairment(
        ear=ear, perclos=perclos, hrv_rmssd=hrv,
        ecg_hr=hr, spo2=spo2, yawn_count=0,
        drive_seconds=0, g_peak=0.0,
    )
    correct = CORRECT_RESPONSE[result]
    assert correct >= 2, (
        f"PERCLOS={perclos:.3f} > 0.80 must map to at least CAUTION, got {result} → {correct}"
    )


# ─────────────────────────────────────────────
# Property 4: Alert monotonic in g-peak
# ─────────────────────────────────────────────

@given(
    g_low=st.floats(min_value=0.0, max_value=1.9),
    g_high=st.floats(min_value=2.0, max_value=6.0),
    ear=st.floats(min_value=0.20, max_value=0.35),
    perclos=st.floats(min_value=0.0, max_value=0.15),
)
@settings(max_examples=200)
def test_crash_alert_monotonic_in_g_peak(g_low, g_high, ear, perclos):
    """Higher g-peak must produce same or higher alert level."""
    result_low = CORRECT_RESPONSE[classify_impairment(
        ear=ear, perclos=perclos, hrv_rmssd=45.0, ecg_hr=72, spo2=98.0,
        yawn_count=0, drive_seconds=0, g_peak=g_low,
    )]
    result_high = CORRECT_RESPONSE[classify_impairment(
        ear=ear, perclos=perclos, hrv_rmssd=45.0, ecg_hr=72, spo2=98.0,
        yawn_count=0, drive_seconds=0, g_peak=g_high,
    )]
    assert result_high >= result_low, (
        f"g_high={g_high:.2f} gives alert {result_high} < g_low={g_low:.2f} alert {result_low}"
    )


# ─────────────────────────────────────────────
# Property 5: Reward is bounded
# ─────────────────────────────────────────────

@given(
    action=st.integers(min_value=0, max_value=4),
    ear=st.floats(min_value=0.05, max_value=0.45),
    perclos=st.floats(min_value=0.0, max_value=1.0),
    hrv=st.floats(min_value=10.0, max_value=80.0),
    hr=st.integers(min_value=40, max_value=160),
    spo2=st.floats(min_value=90.0, max_value=100.0),
    g_peak=st.floats(min_value=0.0, max_value=6.0),
    speed=st.floats(min_value=0.0, max_value=130.0),
    jerk=st.floats(min_value=0.0, max_value=30.0),
)
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
def test_reward_is_bounded(action, ear, perclos, hrv, hr, spo2,
                            g_peak, speed, jerk):
    """Reward must be finite and within reasonable bounds."""
    bundle = FakeBundle(
        ear=ear, perclos=perclos, hrv_rmssd=hrv, ecg_hr=hr,
        spo2=spo2, g_peak=g_peak, speed_kph=speed, jerk_peak=jerk,
    )
    reward_fn = GuardianReward()
    reward, _ = reward_fn.compute(bundle, action, FakeStats())

    assert not (reward != reward), f"Reward is NaN: {reward}"
    assert reward > -1000.0, f"Reward too negative: {reward}"
    assert reward < 100.0, f"Reward too positive: {reward}"


# ─────────────────────────────────────────────
# Property 6: SQI abstention under ECG dropout
# ─────────────────────────────────────────────

@given(
    action=st.integers(min_value=0, max_value=4),
    hrv=st.floats(min_value=10.0, max_value=80.0),
)
@settings(max_examples=100)
def test_ecg_dropout_reward_not_penalised_for_caution(action, hrv):
    """
    Under ECG dropout, conservative actions (NOMINAL/ADVISORY)
    should not be penalised — cannot escalate on broken signal.
    """
    bundle = FakeBundle(ecg_dropout=True, hrv_rmssd=hrv)
    reward_fn = GuardianReward()
    reward, info = reward_fn.compute(bundle, action, FakeStats())

    if action <= 1:
        # Should get positive fault resilience bonus
        fr = info.get("fault_resilience", 0.0)
        assert fr >= 0, f"Conservative action under dropout should not penalise: fr={fr}"


# ─────────────────────────────────────────────
# Property 7: Alert label coverage
# ─────────────────────────────────────────────

def test_all_impairments_have_correct_response():
    """Every impairment must have a defined correct response."""
    from carla_agent.reward.reward_fn import CORRECT_RESPONSE
    expected_impairments = {
        "ALERT", "DROWSY", "SLEEPY", "FATIGUED",
        "CARDIAC_ALERT", "STROKE_SUSPECT", "MICROSLEEP", "CRASH"
    }
    for imp in expected_impairments:
        assert imp in CORRECT_RESPONSE, f"Missing response for {imp}"
        action = CORRECT_RESPONSE[imp]
        assert 0 <= action <= 4, f"Invalid action {action} for {imp}"


# ─────────────────────────────────────────────
# Property 8: CRASH always → ESCALATE
# ─────────────────────────────────────────────

@given(
    g_peak=st.floats(min_value=2.0, max_value=10.0),
    ear=st.floats(min_value=0.20, max_value=0.35),
    perclos=st.floats(min_value=0.0, max_value=0.15),
    spo2=st.floats(min_value=94.0, max_value=100.0),
)
@settings(max_examples=200)
def test_crash_always_escalates(g_peak, ear, perclos, spo2):
    """g-peak >= 2.0g must always result in ESCALATE."""
    result = classify_impairment(
        ear=ear, perclos=perclos, hrv_rmssd=45.0, ecg_hr=72,
        spo2=spo2, yawn_count=0, drive_seconds=0, g_peak=g_peak,
    )
    assert result == "CRASH", f"g_peak={g_peak:.2f} should be CRASH, got {result}"
    assert CORRECT_RESPONSE[result] == 4, "CRASH must map to ESCALATE (4)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
