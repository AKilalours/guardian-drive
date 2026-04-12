"""
server/data_logger.py
Guardian Drive -- Telemetry Logger & Replay System

Logs all sensor streams to JSONL files with timestamps.
Replay mode feeds logged data back through the pipeline for
debugging, regression testing, and offline evaluation.

This implements the full Autonomy Telemetry data lifecycle:
  sensor recording -> timestamped JSONL -> replay -> offline eval

Built by Akila Lourdes Miriyala Francis & Akilan Manivannan
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Generator, Optional


class TelemetryLogger:
    """
    Logs real-time sensor frames to timestamped JSONL files.
    Each line = one pipeline cycle (ECG + EDA + IMU + camera + GPS + scores).
    """

    def __init__(self, log_dir: str = "data/telemetry_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"session_{session_id}.jsonl"
        self._file = open(self.log_path, "a")
        self._frame_count = 0
        print(f"[TelemetryLogger] Logging to {self.log_path}")

    def log_frame(self, frame: dict) -> None:
        """Log a single telemetry frame with timestamp."""
        record = {
            "_ts":          time.time(),
            "_frame":       self._frame_count,
            "level":        frame.get("level", "NOMINAL"),
            "task_a":       frame.get("task_a", {}),
            "task_b":       frame.get("task_b", {}),
            "task_c":       frame.get("task_c", {}),
            "task_d":       frame.get("task_d", {}),
            "sqi":          frame.get("sqi", {}),
            "features":     frame.get("features", {}),
            "webcam":       frame.get("webcam", {}),
            "gps":          frame.get("gps"),
            "vehicle":      frame.get("vehicle", {}),
            "bev":          {"n_objects": frame.get("bev", {}).get("n_objects", 0)
                             if frame.get("bev") else 0},
            "runtime":      frame.get("runtime", {}),
        }
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()
        self._frame_count += 1

    def close(self) -> None:
        self._file.close()
        size_kb = self.log_path.stat().st_size / 1024
        print(f"[TelemetryLogger] Saved {self._frame_count} frames "
              f"({size_kb:.1f} KB) → {self.log_path}")

    @property
    def frame_count(self) -> int:
        return self._frame_count


class TelemetryReplayer:
    """
    Replays logged telemetry sessions for offline analysis.
    Supports real-time, accelerated, and maximum-speed replay.
    """

    def __init__(self, log_path: str, speed: float = 1.0):
        self.log_path = Path(log_path)
        self.speed = speed  # 1.0=real-time, 2.0=2x, 0.0=max speed

    def replay(self) -> Generator[dict, None, None]:
        """Yields frames at original timing (adjusted by speed factor)."""
        frames = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    frames.append(json.loads(line))

        if not frames:
            return

        start_ts    = frames[0]["_ts"]
        replay_start = time.time()

        for frame in frames:
            if self.speed > 0:
                elapsed_original = frame["_ts"] - start_ts
                elapsed_replay   = time.time() - replay_start
                sleep_time = (elapsed_original / self.speed) - elapsed_replay
                if sleep_time > 0:
                    time.sleep(sleep_time)
            yield frame

    def summary(self) -> dict:
        """Summarise a logged session without replaying."""
        frames = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    frames.append(json.loads(line))

        if not frames:
            return {}

        levels = [f.get("level","NOMINAL") for f in frames]
        duration = frames[-1]["_ts"] - frames[0]["_ts"]
        return {
            "session":       self.log_path.name,
            "n_frames":      len(frames),
            "duration_sec":  round(duration, 1),
            "level_counts":  {l: levels.count(l) for l in set(levels)},
            "escalations":   levels.count("ESCALATE"),
            "start":         datetime.fromtimestamp(frames[0]["_ts"]).isoformat(),
        }

    @staticmethod
    def list_sessions(log_dir: str = "data/telemetry_logs") -> list[Path]:
        return sorted(Path(log_dir).glob("session_*.jsonl"))


if __name__ == "__main__":
    print("Guardian Drive — Telemetry Logger Demo\n")

    logger = TelemetryLogger()
    for i in range(10):
        logger.log_frame({
            "level":   "NOMINAL" if i < 7 else "ADVISORY",
            "task_a":  {"hr_bpm": 72+i, "cls": "normal"},
            "task_b":  {"score": round(i*0.05, 2)},
            "task_c":  {"g_peak": 0.1},
            "sqi":     {"overall": 0.95},
            "features":{"hr_bpm": 72+i},
            "vehicle": {"speed_kph": 65+i},
        })
        time.sleep(0.05)
    logger.close()

    print("\nReplaying at 4x speed...")
    replayer = TelemetryReplayer(logger.log_path, speed=4.0)
    for frame in replayer.replay():
        print(f"  frame={frame['_frame']:2d}  level={frame['level']:8s}  "
              f"hr={frame['task_a'].get('hr_bpm','?')}  "
              f"b_score={frame['task_b'].get('score','?')}")

    print("\nSession summary:")
    s = replayer.summary()
    for k,v in s.items():
        print(f"  {k}: {v}")
