import time
import numpy as np
import cv2
import mediapipe as mp

# Simple PERCLOS-like heuristic using FaceMesh eye landmarks
# Returns a continuous score [0,1].
class WebcamDrowsinessMonitor:
    def __init__(self, cam_index=0):
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        self.face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, refine_landmarks=True, max_num_faces=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.closed_hist = []  # 1=eye closed, 0=open
        self.hist_sec = 30.0
        self.last_t = time.time()

        # EAR-ish thresholds (tune on your face)
        self.eye_closed_thresh = 0.18

    def _eye_aspect_ratio(self, lm, idx):
        # idx: 6 points around one eye (mediapipe face mesh indices)
        # We'll use a simplified ratio: vertical / horizontal
        p = np.array([[lm[i].x, lm[i].y] for i in idx], dtype=np.float32)
        # horizontal: corners
        horiz = np.linalg.norm(p[0] - p[3]) + 1e-6
        # vertical: average of two vertical pairs
        vert = (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / 2.0
        return float(vert / horiz)

    def step(self):
        ok, frame = self.cap.read()
        if not ok:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = self.face.process(rgb)
        if not out.multi_face_landmarks:
            return {"score": 0.0, "confidence": 0.0, "note": "no_face"}

        lm = out.multi_face_landmarks[0].landmark

        # Mediapipe indices (approx): left and right eye contours
        left_eye  = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]

        ear_l = self._eye_aspect_ratio(lm, left_eye)
        ear_r = self._eye_aspect_ratio(lm, right_eye)
        ear = (ear_l + ear_r) / 2.0

        closed = 1 if ear < self.eye_closed_thresh else 0

        now = time.time()
        self.closed_hist.append((now, closed))

        # prune history
        cutoff = now - self.hist_sec
        while self.closed_hist and self.closed_hist[0][0] < cutoff:
            self.closed_hist.pop(0)

        # PERCLOS: fraction of time eyes closed in last hist window
        if len(self.closed_hist) < 10:
            perclos = 0.0
            conf = 0.3
        else:
            perclos = sum(c for _, c in self.closed_hist) / len(self.closed_hist)
            conf = 0.8

        # map perclos to score
        score = float(np.clip((perclos - 0.15) / 0.35, 0.0, 1.0))
        return {"score": score, "confidence": conf, "ear": ear, "perclos": perclos}

    def close(self):
        try: self.cap.release()
        except: pass
