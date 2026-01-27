from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from rich.console import Console

from guardian_drive.models.physio_cnn import PhysioCNN1D
from guardian_drive.utils.config import ensure_dir, load_yaml

console = Console()


def _percentile(x, q):
    return float(np.percentile(np.asarray(x, dtype=np.float64), q))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out_dir = ensure_dir(cfg["run"]["out_dir"])
    device = cfg["run"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]CUDA requested but not available. Falling back to CPU.[/yellow]")
        device = "cpu"

    ckpt_path = out_dir / "model_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}. Run `make train` first.")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = PhysioCNN1D(
        in_channels=int(cfg["model"]["in_channels"]),
        hidden_channels=list(cfg["model"]["hidden_channels"]),
        kernel_size=int(cfg["model"]["kernel_size"]),
        dropout=float(cfg["model"]["dropout"]),
        num_classes=1,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    sr = int(cfg["preprocess"]["resample_hz"])
    win = float(cfg["preprocess"]["window_sec"])
    T = int(round(sr * win))
    C = int(cfg["model"]["in_channels"])
    bs = int(cfg["bench"]["batch_size"])

    x = torch.randn((bs, C, T), device=device, dtype=torch.float32)

    # Warmup
    with torch.no_grad():
        for _ in range(int(cfg["bench"]["num_warmup"])):
            _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(int(cfg["bench"]["num_runs"])):
            t0 = time.perf_counter()
            _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms

    p50 = _percentile(times, 50)
    p95 = _percentile(times, 95)
    mean_ms = float(np.mean(times))

    # Model size
    n_params = sum(p.numel() for p in model.parameters())
    bytes_params = sum(p.numel() * p.element_size() for p in model.parameters())
    ckpt_bytes = ckpt_path.stat().st_size

    report = {
        "device": device,
        "input_shape": [bs, C, T],
        "num_runs": int(cfg["bench"]["num_runs"]),
        "latency_ms": {"p50": p50, "p95": p95, "mean": mean_ms},
        "throughput_samples_per_s": float(1000.0 * bs / mean_ms),
        "model": {
            "params": int(n_params),
            "param_bytes": int(bytes_params),
            "checkpoint_bytes": int(ckpt_bytes),
        },
    }

    (out_dir / "bench_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    console.print(report)
    console.print(f"Wrote: {out_dir / 'bench_report.json'}")


if __name__ == "__main__":
    main()
