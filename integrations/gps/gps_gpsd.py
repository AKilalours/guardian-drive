"""GPS via gpsd (OPTIONAL).

Requires:
  pip install gpsd-py3

Usage:
  from integrations.gps.gps_gpsd import GPSDGps
  gps = GPSDGps()
  fix = gps.read_fix()
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from ..base import GPSProvider, GpsFix, GeoPoint

@dataclass
class GPSDGps(GPSProvider):
    host: str = "127.0.0.1"
    port: int = 2947

    def __post_init__(self):
        import gpsd  # type: ignore
        gpsd.connect(host=self.host, port=self.port)
        self._gpsd = gpsd

    def read_fix(self) -> Optional[GpsFix]:
        p = self._gpsd.get_current()
        if p is None:
            return None
        try:
            lat = float(p.lat)
            lon = float(p.lon)
        except Exception:
            return None
        spd = getattr(p, "hspeed", None)
        trk = getattr(p, "track", None)
        err = getattr(p, "error", None)
        return GpsFix(
            point=GeoPoint(lat=lat, lon=lon),
            speed_mps=float(spd) if spd is not None else None,
            heading_deg=float(trk) if trk is not None else None,
            timestamp_unix=time.time(),
            accuracy_m=float(err) if err is not None else None,
        )
