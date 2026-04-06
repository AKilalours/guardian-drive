from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Any, Deque, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    import mediapipe as mp  # type: ignore
except Exception:
    mp = None


@dataclass
class WebcamMetrics:
    available: bool
    face_detected: bool = False
    eyes_closed: Optional[bool] = None
    ear: Optional[float] = None
    perclos_30s: Optional[float] = None
    blink_rate_30s: Optional[float] = None
    mouth_open_ratio: Optional[float] = None
    yawn_events_30s: Optional[int] = None
    drowsy_score: Optional[float] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        eyes_txt = None
        if self.eyes_closed is True:
            eyes_txt = "closed"
        elif self.eyes_closed is False:
            eyes_txt = "open"

        drowsy_bool = None
        if self.drowsy_score is not None:
            drowsy_bool = bool(self.drowsy_score >= 0.65)

        return {
            "available": self.available,
            "face_detected": self.face_detected,
            "eyes_closed": self.eyes_closed,
            "ear": self.ear,
            "perclos_30s": self.perclos_30s,
            "blink_rate_30s": self.blink_rate_30s,
            "mouth_open_ratio": self.mouth_open_ratio,
            "yawn_events_30s": self.yawn_events_30s,
            "drowsy_score": self.drowsy_score,
            "note": self.note,
            # UI aliases
            "eyes": eyes_txt,
            "perclos": self.perclos_30s,
            "yawns": self.yawn_events_30s,
            "drowsy": drowsy_bool,
        }


class WebcamMonitor:
    """
    Live webcam runtime with:
    - MJPEG stream
    - face detection
    - eye closure / EAR
    - PERCLOS over rolling 30s
    - blink count over rolling 30s
    - mouth-open ratio
    - yawn count over rolling 30s
    - heuristic webcam-only drowsiness score

    This is a runtime heuristic, not a clinically validated detector.
    """

    ROLLING_SEC = 30.0

    # EAR hysteresis thresholds
    EAR_CLOSE_THR = 0.14
    EAR_OPEN_THR  = 0.17

    # Mouth / yawn thresholds
    MOUTH_OPEN_THR = 0.28
    YAWN_MIN_OPEN_SEC = 0.60
    YAWN_COOLDOWN_SEC = 1.20

    # Blink thresholds
    BLINK_MIN_CLOSED_SEC = 0.06
    BLINK_MAX_CLOSED_SEC = 0.80

    def __init__(self, cam_index: int = 0, width: int = 640, height: int = 480, fps: int = 15):
        if cv2 is None:
            raise RuntimeError("opencv-python not installed")

        self.cam_index = int(cam_index)
        self.width = int(width)
        self.height = int(height)
        self.fps = max(5, int(fps))

        self._cap = cv2.VideoCapture(self.cam_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open webcam index={self.cam_index}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))

        self._lock = threading.Lock()
        self._running = True
        self._last_jpeg: Optional[bytes] = None
        self._last_metrics = WebcamMetrics(available=True, note="starting...")

        self._hist_closed: Deque[Tuple[float, bool]] = deque()
        self._hist_blinks: Deque[float] = deque()
        self._hist_yawns: Deque[float] = deque()
        self._ear_hist: Deque[float] = deque(maxlen=5)
        self._mouth_hist: Deque[float] = deque(maxlen=5)

        self._prev_closed: Optional[bool] = None
        self._closed_start: Optional[float] = None

        self._yawn_open_start: Optional[float] = None
        self._last_yawn_time: float = 0.0

        self._mp_face = None
        if mp is not None and hasattr(mp, "solutions"):
            self._mp_face = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                refine_landmarks=True,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def close(self) -> None:
        self._running = False
        try:
            self._thr.join(timeout=1.0)
        except Exception:
            pass
        try:
            self._cap.release()
        except Exception:
            pass
        try:
            if self._mp_face is not None:
                self._mp_face.close()
        except Exception:
            pass

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._last_jpeg

    def latest_metrics(self) -> WebcamMetrics:
        with self._lock:
            return self._last_metrics

    @staticmethod
    def _dist(a, b) -> float:
        dx = float(a[0] - b[0])
        dy = float(a[1] - b[1])
        return (dx * dx + dy * dy) ** 0.5

    def _cleanup_histories(self, now: float) -> None:
        cutoff = now - self.ROLLING_SEC
        while self._hist_closed and self._hist_closed[0][0] < cutoff:
            self._hist_closed.popleft()
        while self._hist_blinks and self._hist_blinks[0] < cutoff:
            self._hist_blinks.popleft()
        while self._hist_yawns and self._hist_yawns[0] < cutoff:
            self._hist_yawns.popleft()

    def _ear_from_landmarks(self, pts) -> Optional[float]:
        if pts is None:
            return None

        left = [33, 160, 158, 133, 153, 144]
        right = [362, 385, 387, 263, 373, 380]

        def ear(ix):
            p1, p2, p3, p4, p5, p6 = [pts[i] for i in ix]
            return (self._dist(p2, p6) + self._dist(p3, p5)) / (2.0 * self._dist(p1, p4) + 1e-9)

        return float((ear(left) + ear(right)) / 2.0)

    def _mouth_open_ratio(self, pts) -> Optional[float]:
        if pts is None:
            return None

        top = pts[13]
        bottom = pts[14]
        left = pts[78]
        right = pts[308]

        vertical = self._dist(top, bottom)
        horizontal = self._dist(left, right)
        if horizontal <= 1e-6:
            return None
        return float(vertical / horizontal)

    def _smoothed_ear(self, ear: Optional[float]) -> Optional[float]:
        if ear is None:
            return None
        self._ear_hist.append(float(ear))
        return float(np.mean(self._ear_hist))

    def _smoothed_mouth(self, mouth_ratio: Optional[float]) -> Optional[float]:
        if mouth_ratio is None:
            return None
        self._mouth_hist.append(float(mouth_ratio))
        return float(np.mean(self._mouth_hist))

    def _classify_eyes_closed(self, ear_smooth: Optional[float]) -> Optional[bool]:
        if ear_smooth is None:
            return None

        if self._prev_closed is None:
            return bool(ear_smooth < self.EAR_CLOSE_THR)

        if self._prev_closed:
            return bool(ear_smooth < self.EAR_OPEN_THR)
        return bool(ear_smooth < self.EAR_CLOSE_THR)

    def _update_blink_state(self, now: float, eyes_closed: Optional[bool]) -> None:
        if eyes_closed is None:
            self._prev_closed = None
            self._closed_start = None
            return

        if self._prev_closed is None:
            if eyes_closed:
                self._closed_start = now
            self._prev_closed = eyes_closed
            return

        if (not self._prev_closed) and eyes_closed:
            self._closed_start = now

        if self._prev_closed and (not eyes_closed):
            if self._closed_start is not None:
                dur = now - self._closed_start
                if self.BLINK_MIN_CLOSED_SEC <= dur <= self.BLINK_MAX_CLOSED_SEC:
                    self._hist_blinks.append(now)
            self._closed_start = None

        self._prev_closed = eyes_closed

    def _update_yawn_state(self, now: float, mouth_ratio_smooth: Optional[float]) -> None:
        mouth_open = bool(
            mouth_ratio_smooth is not None and mouth_ratio_smooth >= self.MOUTH_OPEN_THR
        )

        if mouth_open:
            if self._yawn_open_start is None:
                self._yawn_open_start = now
            return

        if self._yawn_open_start is None:
            return

        open_dur = now - self._yawn_open_start
        self._yawn_open_start = None

        if open_dur >= self.YAWN_MIN_OPEN_SEC and (now - self._last_yawn_time) >= self.YAWN_COOLDOWN_SEC:
            self._hist_yawns.append(now)
            self._last_yawn_time = now

    def _annotate(self, frame, metrics: WebcamMetrics) -> None:
        if cv2 is None:
            return

        y = 24
        lines = [
            f"face: {'yes' if metrics.face_detected else 'no'}",
            f"eyes: {('closed' if metrics.eyes_closed else 'open') if metrics.eyes_closed is not None else '--'}",
            f"ear: {metrics.ear:.3f}" if metrics.ear is not None else "ear: --",
            f"perclos30: {metrics.perclos_30s:.2f}" if metrics.perclos_30s is not None else "perclos30: --",
            f"blink30: {metrics.blink_rate_30s:.1f}/min" if metrics.blink_rate_30s is not None else "blink30: --",
            f"mouth: {metrics.mouth_open_ratio:.2f}" if metrics.mouth_open_ratio is not None else "mouth: --",
            f"yawns30: {metrics.yawn_events_30s}" if metrics.yawn_events_30s is not None else "yawns30: --",
            f"drowsy: {metrics.drowsy_score:.2f}" if metrics.drowsy_score is not None else "drowsy: --",
            f"note: {metrics.note}",
        ]

        for line in lines:
            cv2.putText(
                frame,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 22

    def _compute_metrics(self, frame_bgr) -> WebcamMetrics:
        if mp is None or self._mp_face is None:
            return WebcamMetrics(
                available=True,
                face_detected=False,
                note="mediapipe unavailable; streaming only",
            )

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._mp_face.process(rgb)
        now = time.time()

        if not res.multi_face_landmarks:
            self._cleanup_histories(now)
            return WebcamMetrics(
                available=True,
                face_detected=False,
                note="face not detected",
            )

        h, w = frame_bgr.shape[:2]
        lm = res.multi_face_landmarks[0].landmark
        pts = [(p.x * w, p.y * h) for p in lm]

        raw_ear = self._ear_from_landmarks(pts)
        raw_mouth = self._mouth_open_ratio(pts)

        ear = self._smoothed_ear(raw_ear)
        mouth_ratio = self._smoothed_mouth(raw_mouth)

        eyes_closed = self._classify_eyes_closed(ear)
        if eyes_closed is not None:
            self._hist_closed.append((now, eyes_closed))

        self._update_blink_state(now, eyes_closed)
        self._update_yawn_state(now, mouth_ratio)
        self._cleanup_histories(now)

        perclos = None
        if self._hist_closed:
            perclos = float(sum(1 for _, c in self._hist_closed if c) / len(self._hist_closed))

        blink_rate = float(len(self._hist_blinks) * 2.0)  # 30s -> per minute
        yawn_count = int(len(self._hist_yawns))

        score_parts = []
        if perclos is not None:
            score_parts.append(float(np.clip((perclos - 0.10) / 0.30, 0.0, 1.0)) * 0.62)
        if mouth_ratio is not None:
            score_parts.append(float(np.clip((mouth_ratio - 0.26) / 0.14, 0.0, 1.0)) * 0.08)
        if yawn_count > 0:
            score_parts.append(min(0.24, 0.08 * yawn_count))
        if eyes_closed is True:
            score_parts.append(0.12)
        if blink_rate < 6.0 and perclos is not None and perclos > 0.10:
            score_parts.append(0.08)

        dscore = float(np.clip(sum(score_parts), 0.0, 1.0)) if score_parts else 0.0

        return WebcamMetrics(
            available=True,
            face_detected=True,
            eyes_closed=eyes_closed,
            ear=None if ear is None else float(ear),
            perclos_30s=perclos,
            blink_rate_30s=float(blink_rate),
            mouth_open_ratio=None if mouth_ratio is None else float(mouth_ratio),
            yawn_events_30s=yawn_count,
            drowsy_score=dscore,
            note="ok",
        )

    def _loop(self) -> None:
        period = 1.0 / float(self.fps)

        while self._running:
            t0 = time.time()
            ok, frame = self._cap.read()

            if not ok:
                with self._lock:
                    self._last_metrics = WebcamMetrics(
                        available=True,
                        face_detected=False,
                        note="camera read failed",
                    )
                time.sleep(0.2)
                continue

            metrics = self._compute_metrics(frame)
            self._annotate(frame, metrics)

            ok2, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
            jpg_bytes = jpg.tobytes() if ok2 else None

            with self._lock:
                self._last_jpeg = jpg_bytes
                self._last_metrics = metrics

            dt = time.time() - t0
            time.sleep(max(0.0, period - dt))
