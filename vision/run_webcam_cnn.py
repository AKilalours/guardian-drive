#!/usr/bin/env python3
"""
Run webcam drowsiness using a TRAINED CNN model (vision/train_cnn.py).

Run:
  python -m vision.run_webcam_cnn --ckpt runs/vision_model.pt

This is still frame-wise. For real driver monitoring, combine with temporal smoothing
(PERCLOS, moving average, or a sequence model).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as e:
    raise SystemExit("mediapipe is required. Install: pip install mediapipe opencv-python") from e

import torch
from torch import nn
from torchvision import models, transforms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--ckpt", type=str, required=True, help="Path to runs/vision_model.pt")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--prob-thresh", type=float, default=0.65, help="drowsy probability threshold")
    return p.parse_args()


def main():
    args = parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    classes = ckpt.get("classes", ["alert", "drowsy"])
    img_sz = int(ckpt.get("img", 224))

    m = models.mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
    m.load_state_dict(ckpt["model"])
    m.eval()

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_sz, img_sz)),
        transforms.ToTensor(),
    ])

    mp_fd = mp.solutions.face_detection
    fd = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.6)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam index={args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("Webcam CNN drowsiness running. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fd.process(rgb)

        status = "NO FACE"
        p_drowsy = 0.0

        if res.detections:
            det = res.detections[0]
            bb = det.location_data.relative_bounding_box
            x = max(0, int(bb.xmin * frame.shape[1]))
            y = max(0, int(bb.ymin * frame.shape[0]))
            w = int(bb.width * frame.shape[1])
            h = int(bb.height * frame.shape[0])
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            crop = frame[y:y+h, x:x+w].copy()

            x_t = tfm(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0)
            with torch.no_grad():
                logits = m(x_t)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            # assume index of "drowsy" is 1 if folder order is alert,drowsy
            # find it by name when possible
            idx_drowsy = classes.index("drowsy") if "drowsy" in classes else 1
            p_drowsy = float(probs[idx_drowsy])

            status = "DROWSY" if p_drowsy >= args.prob_thresh else "ALERT"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        cv2.putText(frame, f"status: {status}  p_drowsy={p_drowsy:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Guardian Drive - Webcam CNN Drowsiness", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
