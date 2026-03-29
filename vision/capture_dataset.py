#!/usr/bin/env python3
"""
Capture a REAL labeled dataset from your webcam.

This is the missing piece people skip:
  - "Train with webcam" is impossible without labels.
  - This script lets YOU label frames in real-time (key presses).

Folders created:
  data/vision/alert/
  data/vision/drowsy/

Keys:
  a  save frame as ALERT
  d  save frame as DROWSY
  q  quit
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2

try:
    import mediapipe as mp

    # This repo uses the legacy MediaPipe *Solutions* API (mp.solutions.*).
    # If `mp.solutions` is missing, you're importing the wrong package or a too-new build.
    if not hasattr(mp, "solutions"):
        raise RuntimeError(
            "MediaPipe `mp.solutions` is missing. Fix: use a clean Python 3.11 env and install "
            "`mediapipe==0.10.11` (or another 0.10.x that still exposes Solutions), plus "
            "`opencv-python<4.12` and `numpy<2`."
        )
except Exception as e:
    raise SystemExit(
        "mediapipe is required and must expose `mp.solutions`. "
        "Install in this env: pip install 'numpy<2' 'opencv-python<4.12' mediapipe==0.10.11"
    ) from e



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--out", type=str, default="data/vision")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--min-conf", type=float, default=0.6)
    p.add_argument("--save-face-only", action="store_true", default=True,
                   help="Save cropped face region if detected (default True)")
    return p.parse_args()


def main():
    args = parse_args()

    out_root = Path(args.out)
    alert_dir = out_root / "alert"
    drowsy_dir = out_root / "drowsy"
    alert_dir.mkdir(parents=True, exist_ok=True)
    drowsy_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_root / "metadata.csv"
    meta_exists = meta_path.exists()
    meta_f = meta_path.open("a", newline="")
    meta_w = csv.writer(meta_f)
    if not meta_exists:
        meta_w.writerow(["ts", "label", "path", "camera", "face_found", "x", "y", "w", "h"])

    mp_fd = mp.solutions.face_detection
    fd = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=args.min_conf)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam index={args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("Dataset capture running.")
    print("Press: a=ALERT, d=DROWSY, q=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fd.process(rgb)

        face_found = False
        bbox = (0, 0, frame.shape[1], frame.shape[0])
        if res.detections:
            det = res.detections[0]
            bb = det.location_data.relative_bounding_box
            x = max(0, int(bb.xmin * frame.shape[1]))
            y = max(0, int(bb.ymin * frame.shape[0]))
            w = int(bb.width * frame.shape[1])
            h = int(bb.height * frame.shape[0])
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            bbox = (x, y, w, h)
            face_found = True
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        cv2.putText(frame, "a=ALERT  d=DROWSY  q=quit", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Capture Dataset", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        label = None
        if key == ord("a"):
            label = "alert"
            out_dir = alert_dir
        elif key == ord("d"):
            label = "drowsy"
            out_dir = drowsy_dir

        if label:
            ts = time.strftime("%Y%m%d_%H%M%S")
            fn = out_dir / f"{label}_{ts}_{int(time.time()*1000)%100000}.jpg"

            if args.save_face_only and face_found:
                x, y, w, h = bbox
                crop = frame[y:y+h, x:x+w].copy()
                cv2.imwrite(str(fn), crop)
            else:
                cv2.imwrite(str(fn), frame)

            x, y, w, h = bbox
            meta_w.writerow([time.time(), label, str(fn), args.camera, int(face_found), x, y, w, h])
            meta_f.flush()
            print(f"Saved {label}: {fn}")

    meta_f.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
