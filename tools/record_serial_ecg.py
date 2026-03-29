"""Record ECG samples from a serial device to CSV.

Assumes the serial device streams one numeric sample per line (µV or ADC units).

Example Arduino output:
  123
  125
  122
  ...

Run:
  pip install -r requirements-integrations.txt
  python tools/record_serial_ecg.py --port /dev/tty.usbserial-XXXX --baud 115200 --out data/raw/ecg_session.csv
"""
from __future__ import annotations

import argparse, csv, time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--label", default="unknown")
    args = ap.parse_args()

    import serial  # type: ignore
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1.0)
    except Exception as e:
        # Helpful diagnostics for macOS/Linux.
        try:
            from serial.tools import list_ports  # type: ignore
            ports = [p.device for p in list_ports.comports()]
        except Exception:
            ports = []
        msg = [f"Could not open serial port: {args.port}", f"Error: {e}"]
        if ports:
            msg.append("Available ports:")
            msg.extend([f"  - {p}" for p in ports])
            msg.append("Tip (macOS): try /dev/cu.usbserial-* or /dev/cu.usbmodem*")
        raise SystemExit("\n".join(msg))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n = 0
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_unix", "sample", "label"])
        while time.time() - t0 < args.seconds:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue
            try:
                v = float(line)
            except Exception:
                continue
            w.writerow([time.time(), v, args.label])
            n += 1
            if n % 500 == 0:
                print(f"Recorded {n} samples...")
    print(f"Done. Wrote {n} samples to {out}")

if __name__ == "__main__":
    main()
