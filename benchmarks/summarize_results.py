"""
benchmarks/summarize_results.py
Guardian Drive -- Print all verified benchmark results

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
"""
import json
from pathlib import Path

results_dir = Path("learned/results")
print("Guardian Drive -- Verified Benchmark Summary")
print("=" * 60)

files = {
    "task_b_loso_improved.json": ("LOSO AUC (honest)",
        lambda d: f"{d['mean_auc']} ± {d['std_auc']}"),
    "task_b_ddp_eval.json": ("DDP 2x T4 AUC",
        lambda d: str(d['best_auc'])),
    "tensorrt_benchmark.json": ("TensorRT FP32",
        lambda d: f"{d['results']['FP32']['median_ms']}ms "
                  f"{d['results']['speedup_fp32']}x"),
    "hrv_cuda_benchmark.json": ("HRV CUDA speedup",
        lambda d: f"{d['speedup']}x vs NumPy"),
    "sqi_ear_cuda_benchmark.json": ("SQI/EAR CUDA speedup",
        lambda d: f"SQI={d['sqi']['speedup']}x "
                  f"EAR={d['ear']['speedup']}x"),
    "diffusion_eval_real.json": ("Diffusion ADE",
        lambda d: f"{d['ade_m']}m on real nuScenes"),
    "slam_real.json": ("Real SLAM",
        lambda d: f"{d['map_points']} map points "
                  f"{d['frames_tracked']}/{d['frames_processed']} tracked"),
    "sfm_real.json": ("Real SfM (COLMAP)",
        lambda d: f"{d['n_3d_points']} 3D points "
                  f"{d['registered_images']}/30 images"),
    "task_a_metrics.json": ("Task A PTBDB AUC",
        lambda d: f"{d['mean_auc']} ± {d['std_auc']}"),
}

for fname, (label, fmt) in files.items():
    p = results_dir / fname
    if p.exists():
        d = json.loads(p.read_text())
        print(f"  {label:30s}: {fmt(d)}")
    else:
        print(f"  {label:30s}: MISSING")
