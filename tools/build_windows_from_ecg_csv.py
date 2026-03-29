"""Build SensorFrame windows from recorded CSV samples.

Input CSV format (from record_serial_ecg.py):
  t_unix,sample,label

Output JSONL:
  each line is SensorFrame.to_json() with ecg array

Run:
  python tools/build_windows_from_ecg_csv.py --csv data/raw/ecg_session.csv --fs 250 --window 30 --step 5 --out runs/ecg_replay.jsonl
"""
from __future__ import annotations

import argparse, csv
from pathlib import Path
import sys

import numpy as np

# When running `python tools/....py`, Python puts `tools/` on sys.path, not the repo root.
# Ensure imports work without requiring users to remember `python -m ...`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from acquisition.models import SensorFrame, TaskLabel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--fs", type=int, default=250)
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--step", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subject", default="real_00")
    ap.add_argument("--session", default="session_ecg")
    args = ap.parse_args()

    rows=[]
    with open(args.csv, "r", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            try:
                rows.append(float(row["sample"]))
            except Exception:
                pass

    x=np.asarray(rows, dtype=float)
    fs=float(args.fs)
    win_n=int(round(args.window*fs))
    step_n=int(round(args.step*fs))
    if x.size < win_n:
        raise SystemExit(f"Not enough samples: have {x.size}, need {win_n}")

    out=Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # crude label: unknown unless you pass another mechanism
    label=TaskLabel.UNKNOWN

    nwin=0
    with out.open("w", encoding="utf-8") as f:
        for start in range(0, x.size - win_n + 1, step_n):
            seg=x[start:start+win_n]
            sf=SensorFrame(
                session_id=args.session,
                subject_id=args.subject,
                window_sec=float(args.window),
                label=label,
                ecg=seg.astype(float),
                eda=None,
                respiration=None,
                accel=None,
                gyro=None,
                temperature=None,
                alcohol=None,
                belt_tension=0.65,
            )
            f.write(sf.to_json()+"\n")
            nwin += 1
    print(f"Wrote {nwin} windows to {out}")

if __name__ == "__main__":
    main()
