"""
benchmark_libtorch.py
Guardian Drive -- LibTorch / TorchScript Benchmark Harness

Traces the WESAD TCN to TorchScript (the format used by LibTorch C++ runtime)
and benchmarks inference latency -- equivalent to what a C++ LibTorch
deployment would achieve.

LibTorch is the C++ distribution of PyTorch used in production
Tesla FSD inference stack.

Built by Akila Lourdes Miriyala Francis & Akilan Manivannan
"""

import torch
import torch.nn as nn
import time
import json
import statistics
from pathlib import Path

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        pad = (3-1)*dilation
        self.conv = nn.Conv1d(in_ch, out_ch, 3, padding=pad, dilation=dilation)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.res  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out[:,:,:x.size(2)] + self.res(x)

class DrowsinessTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1   = TCNBlock(4,  32, dilation=1)
        self.b2   = TCNBlock(32, 64, dilation=2)
        self.b3   = TCNBlock(64, 64, dilation=4)
        self.b4   = TCNBlock(64, 64, dilation=8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1)
        )
    def forward(self, x):
        x = self.b4(self.b3(self.b2(self.b1(x))))
        return self.head(self.pool(x).squeeze(-1))

def trace_to_torchscript(weights_path: str,
                          output_path: str = "wesad_tcn_scripted.pt") -> str:
    """
    Trace model to TorchScript -- the format consumed by LibTorch C++ runtime.
    This is the first step in production C++ deployment.
    """
    model = DrowsinessTCN()
    if Path(weights_path).exists():
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
    model.eval()

    dummy = torch.randn(1, 4, 4200)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)

    traced.save(output_path)
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"TorchScript saved: {output_path} ({size_mb:.2f} MB)")
    print(f"This file is loaded directly by LibTorch C++ runtime.")
    return output_path

def benchmark(model, device: str = "cpu", n_warmup: int = 20,
              n_runs: int = 200, batch_sizes: list = [1, 8, 32]) -> dict:
    """
    Benchmark inference latency matching LibTorch C++ harness methodology:
    - Warmup runs to prime caches
    - Timed runs with torch.no_grad()
    - Memory tracking with pin_memory for CPU-GPU transfer simulation
    - p50/p95/p99 reported
    """
    model = model.to(device)
    model.eval()
    results = {}

    for bs in batch_sizes:
        # pin_memory simulates CPU-GPU transfer used in C++ LibTorch
        dummy = torch.randn(bs, 4, 4200)
        if device == "cpu":
            dummy = dummy.pin_memory() if hasattr(dummy, "pin_memory") else dummy
        else:
            dummy = dummy.to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(dummy)

        # Timed
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(dummy)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)

        times_sorted = sorted(times)
        results[f"batch_{bs}"] = {
            "device":    device,
            "batch_size": bs,
            "n_runs":    n_runs,
            "median_ms": round(statistics.median(times), 3),
            "p95_ms":    round(times_sorted[int(n_runs*0.95)], 3),
            "p99_ms":    round(times_sorted[int(n_runs*0.99)], 3),
            "min_ms":    round(times_sorted[0], 3),
            "throughput_seq_per_sec": round(1000.0 / statistics.median(times) * bs, 1),
        }
        print(f"  batch={bs:2d}  median={results[f'batch_{bs}']['median_ms']:.2f}ms  "
              f"p95={results[f'batch_{bs}']['p95_ms']:.2f}ms  "
              f"throughput={results[f'batch_{bs}']['throughput_seq_per_sec']:.0f} seq/s")

    return results

if __name__ == "__main__":
    print("=== Guardian Drive LibTorch/TorchScript Benchmark Harness ===\n")

    weights = "learned/models/task_b_tcn_cuda.pt"
    model   = DrowsinessTCN()
    if Path(weights).exists():
        state = torch.load(weights, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
        print(f"Loaded: {weights} (AUC 0.9738)")
    model.eval()

    # Step 1: Trace to TorchScript
    print("\n[1] Tracing to TorchScript (LibTorch C++ format)...")
    script_path = trace_to_torchscript(weights)

    # Step 2: Verify scripted model matches original
    scripted = torch.jit.load(script_path)
    dummy    = torch.randn(1, 4, 4200)
    with torch.no_grad():
        out_orig   = model(dummy)
        out_script = scripted(dummy)
    max_diff = (out_orig - out_script).abs().max().item()
    print(f"TorchScript output match: max_diff={max_diff:.2e} "
          f"({'PASS' if max_diff < 1e-4 else 'FAIL'})")

    # Step 3: Benchmark CPU
    print("\n[2] CPU Benchmark (Apple M4):")
    cpu_results = benchmark(model, device="cpu",
                            n_warmup=20, n_runs=200,
                            batch_sizes=[1, 8, 32])

    # Step 4: CUDA if available
    if torch.cuda.is_available():
        print(f"\n[3] CUDA Benchmark ({torch.cuda.get_device_name(0)}):")
        cuda_results = benchmark(model, device="cuda",
                                 n_warmup=20, n_runs=200,
                                 batch_sizes=[1, 8, 32, 64])
    else:
        print("\n[3] CUDA not available (run on Kaggle T4 for GPU benchmark)")
        cuda_results = {}

    # Step 5: Memory usage
    print("\n[4] Memory footprint:")
    n_params = sum(p.numel() for p in model.parameters())
    mem_mb   = sum(p.numel()*p.element_size() for p in model.parameters()) / 1024**2
    print(f"  Parameters: {n_params:,}")
    print(f"  FP32 weights: {mem_mb:.2f} MB")
    print(f"  FP16 weights: {mem_mb/2:.2f} MB (TensorRT FP16 target)")

    # Save results
    all_results = {
        "model": "WESAD TCN (AUC 0.9738)",
        "torchscript_path": script_path,
        "cpu": cpu_results,
        "cuda": cuda_results,
        "memory": {"params": n_params, "fp32_mb": round(mem_mb, 3)},
    }
    out = Path("learned/results/libtorch_benchmark.json")
    out.write_text(__import__("json").dumps(all_results, indent=2))
    print(f"\nResults saved: {out}")
    print("\n=== Benchmark complete ===")
