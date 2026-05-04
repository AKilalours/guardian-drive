"""Unit tests for Signal Quality Index computation."""
import numpy as np
import pytest

def sqi_numpy(window, thresholds=(0.5,0.05,0.1,0.1)):
    q = [min(1.0, float(np.std(window[c]))/thresholds[c])
         for c in range(4)]
    total = 0.5*q[0] + 0.3*q[1] + 0.2*q[2]
    return {"ecg":q[0],"eda":q[1],"resp":q[2],
            "total":min(1.0,total)}

def test_sqi_good_signal():
    window = np.random.randn(4,4200).astype(np.float32)
    window[0] *= 0.8   # ECG
    window[1] *= 0.1   # EDA
    window[3] *= 0.2   # Resp
    out = sqi_numpy(window)
    assert 0 <= out["total"] <= 1.0

def test_sqi_flat_signal():
    window = np.zeros((4,4200), dtype=np.float32)
    out = sqi_numpy(window)
    assert out["total"] == 0.0  # flat = no quality

def test_sqi_abstain_threshold():
    window = np.zeros((4,4200), dtype=np.float32)
    window += np.random.randn(4,4200)*0.001  # tiny noise
    out = sqi_numpy(window)
    assert out["total"] < 0.30  # should trigger abstain

def test_sqi_range():
    for _ in range(10):
        window = np.random.randn(4,4200).astype(np.float32)
        out = sqi_numpy(window)
        assert 0 <= out["total"] <= 1.0

if __name__ == "__main__":
    test_sqi_good_signal()
    test_sqi_flat_signal()
    test_sqi_abstain_threshold()
    test_sqi_range()
    print("All SQI tests passed")
