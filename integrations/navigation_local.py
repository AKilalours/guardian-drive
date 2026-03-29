from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

from .base import NavigationProvider, GpsFix, GeoPoint, RouteAdvisory


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl   = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))


@dataclass
class LocalHospitalNav(NavigationProvider):
    """Offline navigation advisory using a local hospital list.

    This is intentionally *advisory only*. Real routing requires a map/routing provider.
    """

    hospitals_csv: Path
    assumed_speed_mps: float = 13.0

    def _load(self) -> List[Tuple[str, float, float]]:
        rows: List[Tuple[str, float, float]] = []
        if not self.hospitals_csv.exists():
            return rows
        with self.hospitals_csv.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    rows.append((row["name"], float(row["lat"]), float(row["lon"])))
                except Exception:
                    continue
        return rows

    def nearest_er(self, fix: GpsFix) -> Optional[RouteAdvisory]:
        hospitals = self._load()
        if not hospitals:
            return None
        best = None
        for name, lat, lon in hospitals:
            d = _haversine_m(fix.point.lat, fix.point.lon, lat, lon)
            if best is None or d < best[0]:
                best = (d, name, lat, lon)
        assert best is not None
        distance_m, name, lat, lon = best
        speed = fix.speed_mps or self.assumed_speed_mps
        eta = distance_m / max(speed, 2.0)
        return RouteAdvisory(
            destination_name=name,
            destination_point=GeoPoint(lat=lat, lon=lon),
            eta_sec=float(eta),
            distance_m=float(distance_m),
            provider="local_haversine",
            notes="Offline advisory; replace with OSRM/Google for turn-by-turn.",
        )
