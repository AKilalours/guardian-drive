from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from guardian_drive.utils.config import ensure_dir, load_yaml

console = Console()


def _as_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return [str(v) for v in x]
    return [str(x)]


def _unique_keep_order(xs: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _deep_set(d: dict[str, Any], path: list[str], value: Any) -> None:
    cur: Any = d
    for k in path[:-1]:
        if not isinstance(cur, dict):
            raise TypeError(f"Cannot deep-set into non-dict at key {k!r}")
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    if not isinstance(cur, dict):
        raise TypeError("Cannot deep-set into non-dict at leaf parent")
    cur[path[-1]] = value


def _safe_get(d: dict[str, Any], path: list[str], default: Any = float("nan")) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _mean_std(xs: list[float]) -> tuple[float, float]:
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def _run_module(mod: str, config_path: Path) -> None:
    cmd = [sys.executable, "-m", mod, "--config", str(config_path)]
    subprocess.run(cmd, check=True, env=os.environ.copy())


def _resolve_val_subjects(
    all_subjects: list[str],
    base_val_subjects: list[str],
    test_subject: str,
) -> list[str]:
    base_val = _unique_keep_order([str(s) for s in base_val_subjects])
    target_len = len(base_val)

    # Remove collision
    val = [s for s in base_val if s != test_subject]

    if len(val) == target_len:
        return val

    # Fill back to target length
    candidates = [s for s in all_subjects if s != test_subject and s not in val]
    for c in candidates:
        val.append(c)
        if len(val) == target_len:
            break

    if len(val) != target_len:
        raise ValueError(
            f"Cannot resolve val subjects for test={test_subject}. "
            f"Need {target_len}, got {len(val)}. base_val={base_val} all={all_subjects}"
        )

    return val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", required=True)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--force_rerun", action="store_true")
    args = ap.parse_args()

    base = load_yaml(args.base_config)

    all_subjects = [str(s) for s in _as_list(base.get("data", {}).get("subjects"))]
    all_subjects = [s for s in all_subjects if s != "S12"]
    if not all_subjects:
        raise ValueError("base_config missing data.subjects (or empty).")

    fold_subjects = [str(s) for s in (_as_list(args.subjects) if args.subjects else all_subjects)]

    split_cfg = base.get("data", {}).get("split", {}) if isinstance(base.get("data", {}), dict) else {}
    base_val_subjects = _as_list(
        split_cfg.get("val_subjects", split_cfg.get("val_subject", split_cfg.get("val", None)))
    )
    base_val_subjects = [s for s in base_val_subjects if s]
    if not base_val_subjects:
        raise ValueError(
            "base_config missing data.split.val_subjects/val_subject/val."
        )

    out_root = ensure_dir(Path("reports/wesad_physio/loso"))
    fold_rows: list[dict[str, Any]] = []

    for test_s in fold_subjects:
        out_dir = out_root / f"test_{test_s}"
        cfg_path = out_dir / "config_fold.json"
        report_path = out_dir / "test_report.json"

        if report_path.exists() and not args.force_rerun:
            console.print(f"[SKIP] {test_s} already completed -> {report_path}")
            rep = json.loads(report_path.read_text())
            fold_rows.append(rep)
            continue

        ensure_dir(out_dir)

        cfg = deepcopy(base)
        _deep_set(cfg, ["run", "out_dir"], str(out_dir))
        _deep_set(cfg, ["data", "split", "test_subject"], str(test_s))

        # Resolve and write val subjects into ALL likely keys
        val_subjects = _resolve_val_subjects(all_subjects, base_val_subjects, str(test_s))
        _deep_set(cfg, ["data", "split", "val_subjects"], val_subjects)
        _deep_set(cfg, ["data", "split", "val_subject"], val_subjects)
        _deep_set(cfg, ["data", "split", "val"], val_subjects)

        train_subjects = [s for s in all_subjects if s != test_s and s not in set(val_subjects)]
        _deep_set(cfg, ["data", "split", "train_subjects"], train_subjects)

        if args.epochs is not None:
            _deep_set(cfg, ["train", "epochs"], int(args.epochs))

        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        console.print(f"[FOLD] test={test_s} val={val_subjects} train_n={len(train_subjects)}")

        _run_module("guardian_drive.train.train_physio", cfg_path)
        _run_module("guardian_drive.eval.eval_physio", cfg_path)

        rep = json.loads(report_path.read_text())
        fold_rows.append(rep)

    covs = [float(_safe_get(r, ["coverage_accepted"])) for r in fold_rows]
    prs_raw = [float(_safe_get(r, ["accepted_metrics", "raw", "pr_auc"])) for r in fold_rows]
    f1_raw = [float(_safe_get(r, ["accepted_metrics", "raw", "f1_opt", "f1"])) for r in fold_rows]
    f1_cal_val = [float(_safe_get(r, ["accepted_metrics", "cal", "f1_opt", "f1"])) for r in fold_rows]
    f1_cal_adapt = [float(_safe_get(r, ["accepted_metrics", "cal_adapt", "f1_opt_adapt", "f1"])) for r in fold_rows]

    summary = {
        "n_folds": int(len(fold_rows)),
        "coverage_mean_std": _mean_std(covs),
        "pr_auc_raw_mean_std": _mean_std(prs_raw),
        "f1_raw_mean_std": _mean_std(f1_raw),
        "f1_cal_val_mean_std": _mean_std(f1_cal_val),
        "f1_cal_adapt_mean_std": _mean_std(f1_cal_adapt),
    }

    (out_root / "loso_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    tbl = Table(title="LOSO Summary")
    tbl.add_column("metric")
    tbl.add_column("mean")
    tbl.add_column("std")

    def add_row(name: str, ms: tuple[float, float]) -> None:
        mean, std = ms
        tbl.add_row(name, f"{mean:.4f}", f"{std:.4f}")

    add_row("coverage", summary["coverage_mean_std"])
    add_row("PR-AUC(raw)", summary["pr_auc_raw_mean_std"])
    add_row("F1(raw)", summary["f1_raw_mean_std"])
    add_row("F1(cal@val)", summary["f1_cal_val_mean_std"])
    add_row("F1(cal@adapt)", summary["f1_cal_adapt_mean_std"])
    console.print(tbl)

    console.print(f"Wrote: {out_root / 'loso_summary.json'}")


if __name__ == "__main__":
    main()
