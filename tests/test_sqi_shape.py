import numpy as np
from guardian_drive.quality.sqi import compute_sqi

def test_sqi_shape():
    X = np.random.randn(8, 4, 1000).astype(np.float32)
    q = compute_sqi(X, ["ECG", "EDA", "RESP", "Temp"], rules={"ecg_min_std": 0.0})
    assert q.shape == (8,)
    assert np.all((q >= 0.0) & (q <= 1.0))

