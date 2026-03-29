from sqi.window_quality import compute_window_quality


def test_window_quality_nonempty():
    samples = [0.0, 0.01, 0.03, 0.01, -0.02, 0.0, 0.02]
    out = compute_window_quality(samples)
    assert "overall_score" in out
    assert "abstain" in out
