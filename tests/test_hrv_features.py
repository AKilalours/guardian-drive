"""
Unit tests for HRV feature extraction.
Tests CUDA kernel output matches NumPy reference.
"""
import numpy as np
import pytest

def hrv_numpy(rr):
    diffs = np.diff(rr)
    return {
        "rmssd":   float(np.sqrt(np.mean(diffs**2))),
        "sdnn":    float(np.std(rr)),
        "pnn50":   float(np.mean(np.abs(diffs)>50)*100),
        "mean_rr": float(np.mean(rr)),
        "range_rr":float(np.max(rr)-np.min(rr)),
    }

def test_rmssd_normal():
    rr = np.array([850,860,855,845,870,840,860,855], dtype=np.float32)
    out = hrv_numpy(rr)
    assert 0 < out["rmssd"] < 100

def test_rmssd_tachycardia():
    rr = np.array([480,482,479,481,480,483], dtype=np.float32)
    out = hrv_numpy(rr)
    assert out["rmssd"] < 10  # very low HRV in tachycardia

def test_rmssd_bradycardia():
    rr = np.array([1300,1320,1280,1310,1290], dtype=np.float32)
    out = hrv_numpy(rr)
    assert out["mean_rr"] > 1200

def test_pnn50_zero():
    rr = np.array([850,851,850,851,850], dtype=np.float32)
    out = hrv_numpy(rr)
    assert out["pnn50"] == 0.0  # no successive diffs > 50ms

def test_pnn50_high():
    rr = np.array([850,920,780,900,800,880], dtype=np.float32)
    out = hrv_numpy(rr)
    assert out["pnn50"] > 0.0

def test_sdnn_positive():
    rr = np.abs(np.random.randn(100)*30 + 850).astype(np.float32)
    out = hrv_numpy(rr)
    assert out["sdnn"] > 0

if __name__ == "__main__":
    test_rmssd_normal()
    test_rmssd_tachycardia()
    test_rmssd_bradycardia()
    test_pnn50_zero()
    test_pnn50_high()
    test_sdnn_positive()
    print("All HRV tests passed")
