from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from .base import GPSProvider, GpsFix, GeoPoint


@dataclass
class MockGPS(GPSProvider):
    """Deterministic GPS for simulation / tests."""

    lat: float = 40.7440
    lon: float = -74.0324
    speed_mps: float = 12.0
    heading_deg: float = 90.0
    accuracy_m: float = 8.0

    def get_fix(self) -> Optional[GpsFix]:
        return GpsFix(
            point=GeoPoint(lat=self.lat, lon=self.lon),
            speed_mps=self.speed_mps,
            heading_deg=self.heading_deg,
            timestamp_unix=time.time(),
            accuracy_m=self.accuracy_m,
        )
