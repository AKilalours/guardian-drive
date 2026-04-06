"""
integrations/voice_alerts.py
Guardian Drive — Voice Alert System (macOS say command)

Fires non-blocking voice alerts on policy level changes.
Uses macOS built-in Samantha voice — no dependencies.

Usage:
    from integrations.voice_alerts import VoiceAlerts
    va = VoiceAlerts()
    va.on_level_change("CAUTION", poi_name="Blue Bottle Coffee", poi_dist_mi=0.4)
"""
from __future__ import annotations
import subprocess
import threading
import time
import os
from typing import Optional


class VoiceAlerts:
    """
    Non-blocking voice alerts via macOS `say` command.
    Respects cooldown so it doesn't repeat every 4 seconds.
    """

    COOLDOWN_SEC = 25.0   # minimum seconds between same-level alerts
    ESCALATE_REPEAT = 60.0  # repeat escalate every 60s while active

    SCRIPTS = {
        "ADVISORY": [
            "Heads up. Early fatigue signs detected. Consider a short break soon.",
            "Stay alert. Your fatigue level is rising. Plan a rest stop soon.",
        ],
        "CAUTION": [
            "Warning. Fatigue confirmed. Please find a rest stop within the next few miles.",
            "Caution. Significant drowsiness detected. A rest stop is strongly recommended.",
        ],
        "PULLOVER": [
            "Alert. Severe drowsiness detected. Pull over now and rest immediately.",
            "Pull over now. Your drowsiness level is dangerous. Stop at the nearest safe location.",
        ],
        "ESCALATE": [
            "Emergency. Medical event detected. Routing to nearest hospital. Emergency contacts have been notified.",
            "Emergency protocol active. Pulling over and routing to emergency care. Stay calm.",
        ],
        "NOMINAL": [],  # no voice for nominal
    }

    def __init__(self, voice: str = "Samantha", rate: int = 155, enabled: bool = True):
        self.voice   = voice
        self.rate    = rate
        self.enabled = enabled and self._say_available()
        self._lock   = threading.Lock()
        self._last_level: Optional[str] = None
        self._last_ts: float = 0.0
        self._script_idx: dict[str, int] = {}

        if self.enabled:
            print(f"[voice] macOS voice alerts enabled (voice={voice})")
        else:
            print("[voice] Voice alerts disabled (macOS `say` not available)")

    @staticmethod
    def _say_available() -> bool:
        try:
            r = subprocess.run(["which", "say"], capture_output=True, timeout=2)
            return r.returncode == 0
        except Exception:
            return False

    def on_level_change(
        self,
        level: str,
        poi_name: Optional[str] = None,
        poi_dist_mi: Optional[float] = None,
        route_name: Optional[str] = None,
    ) -> None:
        """Call this every pipeline cycle. Handles cooldown internally."""
        if not self.enabled:
            return

        now = time.time()
        with self._lock:
            same = (level == self._last_level)
            cooldown = self.ESCALATE_REPEAT if level == "ESCALATE" else self.COOLDOWN_SEC
            if same and (now - self._last_ts) < cooldown:
                return
            self._last_level = level
            self._last_ts = now

        scripts = self.SCRIPTS.get(level, [])
        if not scripts:
            return

        # Rotate through scripts so it doesn't repeat identically
        idx = self._script_idx.get(level, 0) % len(scripts)
        self._script_idx[level] = idx + 1
        msg = scripts[idx]

        # Append POI info if available
        if poi_name and poi_dist_mi and level in ("ADVISORY", "CAUTION"):
            msg += f" There is a {poi_name} in {poi_dist_mi} miles."
        elif poi_name and level == "PULLOVER":
            msg += f" The nearest rest stop is {poi_name}."
        elif route_name and level == "ESCALATE":
            msg += f" Routing to {route_name}."

        self._speak(msg)

    def speak(self, message: str) -> None:
        """Speak any message immediately (bypasses cooldown)."""
        if self.enabled:
            self._speak(message)

    def _speak(self, msg: str) -> None:
        def _run():
            try:
                subprocess.run(
                    ["say", "-v", self.voice, "-r", str(self.rate), msg],
                    timeout=15,
                    capture_output=True,
                )
            except Exception:
                pass
        threading.Thread(target=_run, daemon=True).start()


# ── singleton ──────────────────────────────────────────────────────────────────
_instance: Optional[VoiceAlerts] = None

def get_voice() -> VoiceAlerts:
    global _instance
    if _instance is None:
        enabled = os.getenv("GD_VOICE", "1").strip().lower() in {"1","true","yes","on"}
        _instance = VoiceAlerts(enabled=enabled)
    return _instance


if __name__ == "__main__":
    print("Testing Guardian Drive voice alerts…")
    va = VoiceAlerts()
    va.speak("Guardian Drive voice alert system online.")
    time.sleep(3)
    va.on_level_change("ADVISORY", poi_name="Blue Bottle Coffee", poi_dist_mi=0.4)
    time.sleep(5)
    va.on_level_change("CAUTION")
    time.sleep(6)
    va.on_level_change("ESCALATE", route_name="SF General Hospital")
    time.sleep(8)
    print("Done.")
