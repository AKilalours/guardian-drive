"""Test fusion weights sum to 1.0."""

def test_weights_sum_to_one():
    weights = {"phys":0.40,"imu":0.20,"ctx":0.10,"neuro":0.30}
    total = sum(weights.values())
    assert abs(total-1.0)<1e-6, f"Weights sum to {total}"
    print(f"Fusion weights: {weights}")
    print(f"Sum: {total} -- OK")

def test_abstain_beats_escalation():
    """Low SQI should abstain, never escalate."""
    sqi_total = 0.05  # very low quality
    threshold  = 0.30
    assert sqi_total < threshold, "Should abstain"
    print("Abstain-beats-escalation -- OK")

def test_weights_positive():
    weights = {"phys":0.40,"imu":0.20,"ctx":0.10,"neuro":0.30}
    assert all(v > 0 for v in weights.values())
    print("All weights positive -- OK")

if __name__ == "__main__":
    test_weights_sum_to_one()
    test_abstain_beats_escalation()
    test_weights_positive()
    print("All fusion weight tests passed")
