"""Unit tests for Eye Aspect Ratio computation."""
import numpy as np
import pytest

def ear_numpy(landmarks):
    p2p6 = np.linalg.norm(landmarks[1]-landmarks[5])
    p3p5 = np.linalg.norm(landmarks[2]-landmarks[4])
    p1p4 = np.linalg.norm(landmarks[0]-landmarks[3])
    return float((p2p6+p3p5)/(2*p1p4+1e-6))

def test_open_eye():
    # Wide open eye -- high EAR
    lm = np.array([[0,0],[1,2],[2,2],[3,0],[2,-2],[1,-2]],
                   dtype=np.float32)
    assert ear_numpy(lm) > 0.2

def test_closed_eye():
    # Nearly closed -- low EAR
    lm = np.array([[0,0],[1,0.05],[2,0.05],[3,0],[2,-0.05],[1,-0.05]],
                   dtype=np.float32)
    assert ear_numpy(lm) < 0.18

def test_ear_positive():
    lm = np.random.randn(6,2).astype(np.float32)
    assert ear_numpy(lm) >= 0

def test_ear_range():
    for _ in range(20):
        lm = np.random.randn(6,2).astype(np.float32)
        e = ear_numpy(lm)
        assert 0 <= e < 10

if __name__ == "__main__":
    test_open_eye(); test_closed_eye()
    test_ear_positive(); test_ear_range()
    print("All EAR tests passed")
