"""
Webcam-based drowsiness detection using FaceMesh geometry (no training required).

Signals:
  - EAR (Eye Aspect Ratio) -> blink / eye-closure
  - PERCLOS over a sliding window -> drowsiness
  - MAR (Mouth Aspect Ratio) -> yawning (optional)

This is "real" (runs on a live webcam), but not a learned classifier.
If you want training, see vision/train_cnn.py and vision/capture_dataset.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import time
import math

import numpy as np

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
except Exception as e:  # pragma: no cover
    mp = None
    _MEDIAPIPE_IMPORT_ERROR = e



# FaceMesh landmark indices for eye & mouth geometry (MediaPipe).
# These are widely-used stable indices for the 468-landmark mesh.
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)

# Mouth landmarks for MAR (inner/outer mix; good enough for yawn heuristic)
MOUTH = (61, 81, 13, 311, 291, 178, 14, 402)


def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(pts: Dict[int, np.ndarray], eye_idx: Tuple[int, int, int, int, int, int]) -> float:
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
    """
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in eye_idx)
    num = _euclid(p2, p6) + _euclid(p3, p5)
    den = 2.0 * _euclid(p1, p4)
    if den <= 1e-9:
        return 0.0
    return num / den


def mouth_aspect_ratio(pts: Dict[int, np.ndarray], mouth_idx: Tuple[int, ...]) -> float:
    """
    Simple MAR heuristic:
      vertical = ||top-bottom|| using inner lip points
      horizontal = ||left-right||
      MAR = vertical / horizontal
    """
    # indices chosen above: left corner(61), upper(13), right corner(291), lower(14)
    left = pts[mouth_idx[0]]
    right = pts[mouth_idx[4]]
    upper = pts[mouth_idx[2]]
    lower = pts[mouth_idx[6]]
    vert = _euclid(upper, lower)
    horiz = _euclid(left, right)
    if horiz <= 1e-9:
        return 0.0
    return vert / horiz


@dataclass
class DrowsinessConfig:
    ear_thresh: float = 0.21           # below -> "eyes closed"
    mar_thresh: float = 0.70           # above -> "yawn" (tune per person/camera)
    perclos_window_sec: float = 30.0   # sliding window for PERCLOS
    perclos_thresh: float = 0.40       # >40% closed in window -> drowsy
    min_face_conf: float = 0.5
    min_track_conf: float = 0.5
    max_num_faces: int = 1


@dataclass
class DrowsinessState:
    ear: float = 0.0
    mar: float = 0.0
    perclos: float = 0.0
    eyes_closed: bool = False
    yawn: bool = False
    drowsy: bool = False
    face_found: bool = False
    ts: float = 0.0


class DrowsinessEstimator:
    """
    Maintains a sliding window of eyes-closed events to compute PERCLOS.
    """

    def __init__(self, cfg: Optional[DrowsinessConfig] = None):
        if cfg is None:
            cfg = DrowsinessConfig()
        self.cfg = cfg

        if mp is None:
            raise ImportError(
                "mediapipe is required for webcam drowsiness detection.\n"
                "Install: pip install mediapipe opencv-python\n"
                f"Import error: {_MEDIAPIPE_IMPORT_ERROR}"
            )

        self._mp_face_mesh = mp.solutions.face_mesh
        self._mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.cfg.max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=self.cfg.min_face_conf,
            min_tracking_confidence=self.cfg.min_track_conf,
        )

        # deque of (timestamp, eyes_closed_bool)
        self._closed_hist: Deque[Tuple[float, bool]] = deque()

    def _trim_hist(self, now: float):
        w = self.cfg.perclos_window_sec
        while self._closed_hist and (now - self._closed_hist[0][0]) > w:
            self._closed_hist.popleft()

    def update(self, rgb_frame: np.ndarray) -> DrowsinessState:
        """
        Update estimator with an RGB frame (H,W,3) uint8.
        """
        st = DrowsinessState(ts=time.time())

        res = self._mesh.process(rgb_frame)
        if not res.multi_face_landmarks:
            st.face_found = False
            # still trim window to avoid stale
            self._trim_hist(st.ts)
            st.perclos = self._compute_perclos()
            st.drowsy = st.perclos >= self.cfg.perclos_thresh
            return st

        st.face_found = True
        face = res.multi_face_landmarks[0]
        h, w, _ = rgb_frame.shape

        pts: Dict[int, np.ndarray] = {}
        for i, lm in enumerate(face.landmark):
            pts[i] = np.array([lm.x * w, lm.y * h], dtype=np.float32)

        ear_l = eye_aspect_ratio(pts, LEFT_EYE)
        ear_r = eye_aspect_ratio(pts, RIGHT_EYE)
        st.ear = (ear_l + ear_r) / 2.0
        st.eyes_closed = st.ear < self.cfg.ear_thresh

        try:
            st.mar = mouth_aspect_ratio(pts, MOUTH)
            st.yawn = st.mar > self.cfg.mar_thresh
        except Exception:
            st.mar = 0.0
            st.yawn = False

        # update history, compute perclos
        self._closed_hist.append((st.ts, st.eyes_closed))
        self._trim_hist(st.ts)
        st.perclos = self._compute_perclos()
        st.drowsy = (st.perclos >= self.cfg.perclos_thresh)

        return st

    def _compute_perclos(self) -> float:
        if not self._closed_hist:
            return 0.0
        closed = sum(1 for _, c in self._closed_hist if c)
        return closed / float(len(self._closed_hist))
