"""
learned/cuda_features.py
Guardian Drive -- CUDA-Accelerated Feature Extraction

Replaces NumPy/Python loops with compiled CUDA kernels when available.
Falls back to NumPy automatically when running on CPU (Mac demo mode).

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
"""
import numpy as np
import torch
import importlib.util
import glob
import os
from pathlib import Path

# ── Auto-detect compiled CUDA kernels ─────────────────────────────
_HRV_EXT  = None
_SQI_EXT  = None
_BEV_EXT  = None
CUDA_AVAILABLE = torch.cuda.is_available()

def _load_ext(name, search_dirs):
    """Load compiled .so extension if available."""
    for d in search_dirs:
        matches = glob.glob(f"{d}/{name}*.so")
        if matches:
            spec = importlib.util.spec_from_file_location(name, matches[0])
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None

def init_cuda_kernels(search_dirs=None):
    """Initialize CUDA kernels from compiled .so files."""
    global _HRV_EXT, _SQI_EXT, _BEV_EXT
    if not CUDA_AVAILABLE:
        print("[CUDAFeatures] No CUDA -- using NumPy fallback")
        return
    dirs = search_dirs or [
        "/kaggle/working/hrv_kernel",
        "/kaggle/working/sqi_kernel",
        "/kaggle/working/bev_kernel",
        "cuda_kernels",
        ".",
    ]
    _HRV_EXT = _load_ext("hrv_ext", dirs)
    _SQI_EXT = _load_ext("sqi_ext", dirs)
    _BEV_EXT = _load_ext("bev_ext", dirs)
    print(f"[CUDAFeatures] HRV={'CUDA' if _HRV_EXT else 'NumPy'} "
          f"SQI={'CUDA' if _SQI_EXT else 'NumPy'} "
          f"BEV={'CUDA' if _BEV_EXT else 'PyTorch'}")

# ── HRV features ──────────────────────────────────────────────────
def compute_hrv_features(rr_intervals: np.ndarray) -> dict:
    """
    Compute HRV features from RR intervals.
    Uses CUDA kernel if available (61.7x speedup), else NumPy.
    
    Args:
        rr_intervals: [N] array of RR intervals in ms
    Returns:
        dict with rmssd, sdnn, pnn50, mean_rr, range_rr
    """
    if _HRV_EXT is not None and CUDA_AVAILABLE:
        rr_t = torch.FloatTensor(rr_intervals).unsqueeze(0).cuda()
        N    = len(rr_intervals)
        outs = _HRV_EXT.hrv_features(rr_t, N)
        return {
            "rmssd":   float(outs[0][0].cpu()),
            "sdnn":    float(outs[1][0].cpu()),
            "pnn50":   float(outs[2][0].cpu()),
            "mean_rr": float(outs[3][0].cpu()),
            "range_rr":float(outs[4][0].cpu()),
            "backend": "cuda"
        }
    # NumPy fallback
    diffs = np.diff(rr_intervals)
    return {
        "rmssd":   float(np.sqrt(np.mean(diffs**2))),
        "sdnn":    float(np.std(rr_intervals)),
        "pnn50":   float(np.mean(np.abs(diffs)>50)*100),
        "mean_rr": float(np.mean(rr_intervals)),
        "range_rr":float(np.max(rr_intervals)-np.min(rr_intervals)),
        "backend": "numpy"
    }

# ── SQI computation ───────────────────────────────────────────────
def compute_sqi_cuda(signal: np.ndarray,
                      thresholds=(0.5,0.05,0.1,0.1)) -> dict:
    """
    Compute SQI across 4 channels.
    Uses CUDA kernel if available (73.4x speedup), else Python.
    
    Args:
        signal: [4, T] physiological signal array
        thresholds: per-channel normalization thresholds
    Returns:
        dict with per-channel and total SQI
    """
    if _SQI_EXT is not None and CUDA_AVAILABLE:
        sig_t = torch.FloatTensor(signal).unsqueeze(0).cuda()
        thr_t = torch.FloatTensor(thresholds).cuda()
        out   = _SQI_EXT.sqi(sig_t, thr_t)  # [1, 4]
        q = out[0].cpu().numpy()
        total = float(0.5*q[0]+0.3*q[1]+0.2*q[2]+0.0*q[3])
        return {"ecg":float(q[0]),"eda":float(q[1]),
                "resp":float(q[2]),"motion":float(q[3]),
                "total":round(min(1.0,total),3),"backend":"cuda"}
    # Python fallback
    q = [min(1.0, float(np.var(signal[c]))**0.5/thresholds[c])
         for c in range(4)]
    total = 0.5*q[0]+0.3*q[1]+0.2*q[2]
    return {"ecg":round(q[0],3),"eda":round(q[1],3),
            "resp":round(q[2],3),"motion":round(q[3],3),
            "total":round(min(1.0,total),3),"backend":"numpy"}

# ── EAR computation ───────────────────────────────────────────────
def compute_ear_cuda(landmarks_batch: np.ndarray) -> np.ndarray:
    """
    Compute EAR for a batch of frames.
    Uses CUDA kernel if available (319x speedup), else NumPy.
    
    Args:
        landmarks_batch: [B, 6, 2] eye landmark coordinates
    Returns:
        [B] EAR values
    """
    if _SQI_EXT is not None and CUDA_AVAILABLE:
        lm_t = torch.FloatTensor(landmarks_batch).view(
            len(landmarks_batch), 12).cuda()
        return _SQI_EXT.ear(lm_t).cpu().numpy()
    # NumPy fallback
    out = []
    for lm in landmarks_batch:
        p2p6 = np.linalg.norm(lm[1]-lm[5])
        p3p5 = np.linalg.norm(lm[2]-lm[4])
        p1p4 = np.linalg.norm(lm[0]-lm[3])
        out.append((p2p6+p3p5)/(2*p1p4+1e-6))
    return np.array(out)

# ── BEV projection ────────────────────────────────────────────────
def bev_project_cuda(world_points: np.ndarray,
                      rotation: np.ndarray,
                      translation: np.ndarray,
                      H=200, W=200,
                      res=0.5, x_min=-50., y_min=-50.) -> np.ndarray:
    """
    Project world points to BEV grid.
    Uses CUDA kernel if available, else PyTorch einsum.
    
    Args:
        world_points: [N, 3] world coordinates
        rotation: [3, 3] ego rotation matrix
        translation: [3] ego translation
    Returns:
        [H, W] occupancy grid
    """
    if _BEV_EXT is not None and CUDA_AVAILABLE:
        pts_t = torch.FloatTensor(world_points).cuda()
        R_t   = torch.FloatTensor(rotation).cuda()
        t_t   = torch.FloatTensor(translation).cuda()
        return _BEV_EXT.bev_project(
            pts_t, R_t, t_t, H, W, res, x_min, y_min).cpu().numpy()
    # PyTorch einsum fallback
    pts = torch.FloatTensor(world_points)
    R   = torch.FloatTensor(rotation)
    t   = torch.FloatTensor(translation)
    c   = pts - t.unsqueeze(0)
    e   = torch.einsum('ij,nj->ni', R.T, c)
    gx  = ((e[:,0]-x_min)/res).long().clamp(0,W-1)
    gy  = ((e[:,1]-y_min)/res).long().clamp(0,H-1)
    grid = torch.zeros(H, W)
    grid.index_put_((gy,gx),
                     torch.ones(len(pts)), accumulate=True)
    return grid.numpy()

# Initialize on import
init_cuda_kernels()
