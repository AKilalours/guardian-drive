#!/usr/bin/env python3
"""
Run real-time webcam drowsiness detection.

Usage:
  python -m vision.run_webcam_drowsy --camera 0

Keys:
  q   quit
  s   screenshot (saved under ./runs/webcam/)
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np

from .drowsiness_heuristics import DrowsinessConfig, DrowsinessEstimator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0, help="Webcam device index (default 0)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--ear-thresh", type=float, default=0.21)
    p.add_argument("--mar-thresh", type=float, default=0.70)
    p.add_argument("--window-sec", type=float, default=30.0)
    p.add_argument("--perclos-thresh", type=float, default=0.40)
    p.add_argument("--show", action="store_true", help="Show OpenCV window (default True)", default=True)
    return p.parse_args()


def main():
    args = parse_args()

    cfg = DrowsinessConfig(
        ear_thresh=args.ear_thresh,
        mar_thresh=args.mar_thresh,
        perclos_window_sec=args.window_sec,
        perclos_thresh=args.perclos_thresh,
    )
    est = DrowsinessEstimator(cfg)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam index={args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    out_dir = Path("runs") / "webcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    fps_hist = []
    last = time.time()

    print("Webcam drowsiness running. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed; exiting.")
            break

        # OpenCV gives BGR; estimator expects RGB.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st = est.update(rgb)

        now = time.time()
        dt = now - last
        last = now
        if dt > 0:
            fps_hist.append(1.0 / dt)
        if len(fps_hist) > 30:
            fps_hist = fps_hist[-30:]
        fps = sum(fps_hist) / max(1, len(fps_hist))

        # Overlay
        status = "DROWSY" if st.drowsy else "ALERT"
        if not st.face_found:
            status = "NO FACE"

        lines = [
            f"status: {status}",
            f"EAR: {st.ear:.3f} (th={cfg.ear_thresh:.2f})",
            f"PERCLOS({cfg.perclos_window_sec:.0f}s): {st.perclos:.2f} (th={cfg.perclos_thresh:.2f})",
            f"Yawn: {int(st.yawn)}  MAR: {st.mar:.2f} (th={cfg.mar_thresh:.2f})",
            f"FPS: {fps:.1f}",
        ]

        y = 30
        for ln in lines:
            cv2.putText(frame, ln, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            y += 30

        # Draw a simple colored banner
        banner_color = (0, 0, 255) if st.drowsy else (0, 200, 0)
        if not st.face_found:
            banner_color = (80, 80, 80)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 10), banner_color, thickness=-1)

        cv2.imshow("Guardian Drive - Webcam Drowsiness", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            ts = time.strftime("%Y%m%d_%H%M%S")
            fn = out_dir / f"frame_{ts}.jpg"
            cv2.imwrite(str(fn), frame)
            print(f"Saved {fn}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
