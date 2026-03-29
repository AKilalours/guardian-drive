"""Runtime GPS provider for demo servers.

- Backend stores last known GPS fix in-process (thread-safe).
- Browser dashboard can POST geolocation updates to /api/gps.
- Real GPS devices can be integrated by swapping GPSProvider.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from integrations.base import GPSProvider, GpsFix, GeoPoint

class RuntimeGPS(GPSProvider):
    _lock = threading.Lock()
    _fix: Optional[GpsFix] = None

    @classmethod
    def set_fix(cls, lat: float, lon: float, accuracy_m: float | None = None, timestamp_unix: float | None = None) -> None:
        with cls._lock:
            cls._fix = GpsFix(
                point=GeoPoint(lat=float(lat), lon=float(lon)),
                accuracy_m=None if accuracy_m is None else float(accuracy_m),
                timestamp_unix=timestamp_unix if timestamp_unix is not None else time.time(),
            )

    def read_fix(self) -> Optional[GpsFix]:
        with self._lock:
            return self._fix
