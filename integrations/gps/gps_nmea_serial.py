"""GPS via NMEA over serial (OPTIONAL).

Requires:
  pip install pyserial

Usage example:
  from integrations.gps.gps_nmea_serial import NMEASerialGPS
  gps = NMEASerialGPS(port="/dev/tty.usbserial-XXXX", baud=9600)
  fix = gps.read_fix()
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from ..base import GPSProvider, GpsFix, GeoPoint

@dataclass
class NMEASerialGPS(GPSProvider):
    port: str
    baud: int = 9600
    timeout: float = 1.0

    def __post_init__(self):
        import serial  # type: ignore
        self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout)

    def read_fix(self) -> Optional[GpsFix]:
        # Minimal parser for $GPRMC; robust parsing left for production
        for _ in range(20):
            line = self._ser.readline().decode(errors="ignore").strip()
            if not line.startswith("$GPRMC"):
                continue
            parts = line.split(",")
            if len(parts) < 12 or parts[2] != "A":
                continue
            lat = _parse_lat(parts[3], parts[4])
            lon = _parse_lon(parts[5], parts[6])
            spd_kn = float(parts[7] or 0.0)
            hdg = float(parts[8] or 0.0)
            return GpsFix(
                point=GeoPoint(lat=lat, lon=lon),
                speed_mps=spd_kn * 0.514444,
                heading_deg=hdg,
                timestamp_unix=time.time(),
            )
        return None

def _parse_lat(v: str, hemi: str) -> float:
    # ddmm.mmmm
    if not v:
        return 0.0
    dd = float(v[:2])
    mm = float(v[2:])
    out = dd + mm/60.0
    return -out if hemi.upper() == "S" else out

def _parse_lon(v: str, hemi: str) -> float:
    # dddmm.mmmm
    if not v:
        return 0.0
    dd = float(v[:3])
    mm = float(v[3:])
    out = dd + mm/60.0
    return -out if hemi.upper() == "W" else out
