from features.hrv_live import rr_from_peak_times, compute_hrv


def test_hrv_live_basic():
    rr = rr_from_peak_times([0.0, 1.0, 2.0, 3.0])
    out = compute_hrv(rr)
    assert out["n_rr"] == 3
    assert 59.0 <= out["hr_bpm"] <= 61.0
