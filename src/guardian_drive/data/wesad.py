from __future__ import annotations

import argparse
import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR_CHEST_HZ = 700
SR_WRIST_HZ = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}


@dataclass
class SubjectRaw:
    subject: str
    chest: dict[str, np.ndarray]
    wrist: dict[str, np.ndarray]
    labels: np.ndarray


def find_subject_pickle(wesad_root: str | Path, subject: str) -> Path:
    root = Path(wesad_root)
    candidates = [
        root / subject / f"{subject}.pkl",
        root / subject / f"{subject}.pickle",
        root / f"{subject}.pkl",
        root / f"{subject}.pickle",
    ]
    for c in candidates:
        if c.exists():
            return c
    hits = list(root.glob(f"**/{subject}.pkl")) + list(root.glob(f"**/{subject}.pickle"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Could not find pickle for subject {subject} under {root}")


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def load_subject_raw(
    wesad_root: str | Path,
    subject: str,
    chest_signals: Iterable[str] | None = None,
    wrist_signals: Iterable[str] | None = None,
) -> SubjectRaw:
    pkl = find_subject_pickle(wesad_root, subject)
    data = _load_pickle(pkl)

    signal = data.get("signal")
    if not isinstance(signal, dict):
        raise ValueError(f"Unexpected WESAD pickle structure in {pkl}")

    chest_all = signal.get("chest", {}) if isinstance(signal.get("chest"), dict) else {}
    wrist_all = signal.get("wrist", {}) if isinstance(signal.get("wrist"), dict) else {}

    chest_signals = list(chest_signals or [])
    wrist_signals = list(wrist_signals or [])

    chest: dict[str, np.ndarray] = {}
    for k in chest_signals:
        if k not in chest_all:
            raise KeyError(
                f"Missing chest signal {k} for {subject} (available={list(chest_all.keys())})"
            )
        x = np.asarray(chest_all[k], dtype=np.float32)
        chest[k] = x.reshape(-1) if x.ndim > 1 else x

    wrist: dict[str, np.ndarray] = {}
    for k in wrist_signals:
        if k not in wrist_all:
            raise KeyError(
                f"Missing wrist signal {k} for {subject} (available={list(wrist_all.keys())})"
            )
        x = np.asarray(wrist_all[k], dtype=np.float32)
        wrist[k] = x.reshape(-1) if x.ndim > 1 else x

    labels = np.asarray(data.get("label"), dtype=np.int64).reshape(-1)
    if labels.size == 0:
        raise ValueError(f"No labels found in {pkl}")

    return SubjectRaw(subject=subject, chest=chest, wrist=wrist, labels=labels)


def check_dataset(wesad_root: str | Path, subjects: list[str]) -> int:
    root = Path(wesad_root)
    if not root.exists():
        print(f"[FAIL] Root does not exist: {root}")
        return 2

    ok = True
    for s in subjects:
        try:
            p = find_subject_pickle(root, s)
            data = _load_pickle(p)
            labels = np.asarray(data.get("label"), dtype=np.int64).reshape(-1)
            signal = data.get("signal", {})
            chest = signal.get("chest", {}) if isinstance(signal, dict) else {}
            print(
                f"[OK] {s}: file={p.relative_to(root)} labels={labels.size} chest_keys={list(chest.keys())}"
            )
        except Exception as e:
            ok = False
            print(f"[FAIL] {s}: {e}")

    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="WESAD utilities")
    parser.add_argument("--root", required=True, help="Path to extracted WESAD root")
    parser.add_argument("--check", action="store_true", help="Validate dataset layout")
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=[
            "S2",
            "S3",
            "S4",
            "S5",
            "S6",
            "S7",
            "S8",
            "S9",
            "S10",
            "S11",
            "S12",
            "S13",
            "S14",
            "S15",
            "S16",
        ],
    )
    args = parser.parse_args()

    if args.check:
        raise SystemExit(check_dataset(args.root, list(args.subjects)))

    parser.print_help()


if __name__ == "__main__":
    main()
