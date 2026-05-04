"""
camera_robustness/run_robustness_suite.py
Guardian Drive -- Camera Robustness Test Suite

Tests EAR computation stability under simulated camera conditions:
- Normal lighting
- Low light (increased landmark jitter)
- Glare (overexposure uncertainty)
- Glasses (vertical eye compression)
- Head yaw 30deg (foreshortening)
- Motion blur (temporal averaging)
- Partial occlusion (eyelid compression)

NOTE: Uses synthetic landmark perturbations as proxy for real camera
conditions. Real camera footage required for production validation.

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
"""
import numpy as np
import json
import time
from pathlib import Path


def ear_compute(lm: np.ndarray) -> float:
    """Eye Aspect Ratio from 6 landmarks."""
    p2p6 = np.linalg.norm(lm[1] - lm[5])
    p3p5 = np.linalg.norm(lm[2] - lm[4])
    p1p4 = np.linalg.norm(lm[0] - lm[3])
    return float((p2p6 + p3p5) / (2 * p1p4 + 1e-6))


def generate_open_eye() -> np.ndarray:
    """Standard open eye landmarks (normalized coordinates)."""
    return np.array([
        [0.0,  0.0],   # p1 outer corner
        [0.5,  0.3],   # p2 upper outer
        [1.5,  0.3],   # p3 upper inner
        [2.0,  0.0],   # p4 inner corner
        [1.5, -0.3],   # p5 lower inner
        [0.5, -0.3],   # p6 lower outer
    ], dtype=np.float32)


def simulate_condition(condition: str,
                        base_lm: np.ndarray) -> np.ndarray:
    """Apply synthetic perturbation for camera condition."""
    lm = base_lm.copy()

    if condition == "normal":
        lm += np.random.randn(*lm.shape) * 0.01

    elif condition == "low_light":
        # Low light: detector uncertainty increases landmark jitter
        lm += np.random.randn(*lm.shape) * 0.05

    elif condition == "glare":
        # Glare: overexposure causes larger landmark uncertainty
        lm += np.random.randn(*lm.shape) * 0.08

    elif condition == "glasses":
        # Glasses: slight vertical compression of detected eye region
        lm[1:3, 1] *= 0.7   # upper lids compressed
        lm[4:6, 1] *= 0.7   # lower lids compressed
        lm += np.random.randn(*lm.shape) * 0.02

    elif condition == "head_yaw_30deg":
        # Head turn: horizontal foreshortening
        yaw = np.radians(30)
        lm[:, 0] *= np.cos(yaw)
        lm += np.random.randn(*lm.shape) * 0.02

    elif condition == "motion_blur":
        # Motion blur: landmarks averaged across 5 temporal frames
        blurred = np.mean(
            [base_lm + np.random.randn(*base_lm.shape) * 0.1
             for _ in range(5)],
            axis=0
        ).astype(np.float32)
        lm = blurred

    elif condition == "partial_occlusion":
        # Upper eyelid partially occluded (e.g. by hat brim)
        lm[1:3, 1] *= 0.5
        lm += np.random.randn(*lm.shape) * 0.03

    elif condition == "sunglasses":
        # Sunglasses: eye almost invisible, EAR near zero
        lm[1:3, 1] *= 0.1
        lm[4:6, 1] *= 0.1
        lm += np.random.randn(*lm.shape) * 0.05

    return lm


CONDITIONS = [
    "normal",
    "low_light",
    "glare",
    "glasses",
    "sunglasses",
    "head_yaw_30deg",
    "motion_blur",
    "partial_occlusion",
]

N_TRIALS      = 1000
DROWSY_THRESH = 0.18   # EAR < 0.18 = eye closing


def run_suite() -> dict:
    print("Guardian Drive -- Camera Robustness Suite")
    print("=" * 60)
    print("Method: Synthetic landmark perturbation proxy")
    print("Note:   Real camera footage needed for production validation\n")

    base_lm      = generate_open_eye()
    baseline_ear = ear_compute(base_lm)
    results      = {}

    for condition in CONDITIONS:
        np.random.seed(42)
        ears    = []
        t_start = time.perf_counter()

        for _ in range(N_TRIALS):
            lm  = simulate_condition(condition, base_lm)
            ear = ear_compute(lm)
            ears.append(ear)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        ears       = np.array(ears)

        false_drowsy = float(np.mean(ears < DROWSY_THRESH))
        stability    = float(np.std(ears))
        mean_ear     = float(np.mean(ears))
        ear_drift    = abs(mean_ear - baseline_ear)
        p50_ms       = elapsed_ms / N_TRIALS

        results[condition] = {
            "mean_ear":              round(mean_ear, 4),
            "std_ear":               round(stability, 4),
            "ear_drift_from_normal": round(ear_drift, 4),
            "false_drowsy_rate":     round(false_drowsy, 4),
            "p50_ms_per_frame":      round(p50_ms, 4),
            "n_trials":              N_TRIALS,
            "pass":                  false_drowsy < 0.10,
        }

        status = "PASS" if false_drowsy < 0.10 else "FAIL (high FP)"
        print(f"  {condition:25s}  "
              f"EAR={mean_ear:.3f}±{stability:.3f}  "
              f"FP={false_drowsy:.3f}  {status}")

    print(f"\nBaseline EAR (normal open eye): {baseline_ear:.4f}")
    print(f"Drowsy threshold:               {DROWSY_THRESH}")

    final = {
        "suite":          "Camera Robustness -- EAR Stability",
        "method":         "Synthetic landmark perturbation proxy",
        "real_footage":   "Required for production validation",
        "drowsy_threshold": DROWSY_THRESH,
        "n_trials":       N_TRIALS,
        "baseline_ear":   round(baseline_ear, 4),
        "conditions":     results,
        "summary": {
            "passed": sum(1 for v in results.values() if v["pass"]),
            "failed": sum(1 for v in results.values() if not v["pass"]),
            "total":  len(results),
        },
        "authors": "Akilan Manivannan & Akila Lourdes Miriyala Francis",
    }

    Path("camera_robustness/results").mkdir(parents=True, exist_ok=True)
    Path("learned/results").mkdir(parents=True, exist_ok=True)

    out = Path("camera_robustness/results/ear_robustness.json")
    out.write_text(json.dumps(final, indent=2))
    Path("learned/results/camera_robustness.json").write_text(
        json.dumps(final, indent=2))

    print(f"\nSaved: {out}")
    print(f"Passed: {final['summary']['passed']}/{final['summary']['total']}")
    return final


if __name__ == "__main__":
    run_suite()
