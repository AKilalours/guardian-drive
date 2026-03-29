from __future__ import annotations

"""Online hospital lookup using OpenStreetMap Overpass API.

Why this exists:
- Your local hospitals.csv is a demo placeholder.
- Recruiter-grade "real-world" routing needs a real POI source.

This provider:
1) queries Overpass for nearby hospitals/emergency clinics
2) picks the closest
3) returns a RouteAdvisory (still advisory-only; no vehicle control)

Notes:
- Overpass is rate-limited; we cache results for (lat,lon) buckets.
- If Overpass fails (offline / rate limit), you should fall back to LocalHospitalNav.
"""

import json
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import requests

from .base import NavigationProvider, GpsFix, GeoPoint, RouteAdvisory


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


@dataclass
class OSMHospitalNav(NavigationProvider):
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    radius_m: int = 8000
    assumed_speed_mps: float = 13.0
    timeout_s: float = 8.0
    cache_ttl_s: float = 60.0

    # cache key: (lat_bucket, lon_bucket, radius_m) -> (timestamp, best)
    _cache: Dict[Tuple[int, int, int], Tuple[float, Optional[Tuple[str, float, float]]]] = field(default_factory=dict)

    def _bucket(self, lat: float, lon: float) -> Tuple[int, int]:
        # ~0.01 deg ~ 1.1km; reduces Overpass spam.
        return (int(lat * 100), int(lon * 100))

    def nearest_er(self, fix: GpsFix) -> Optional[RouteAdvisory]:
        key = (*self._bucket(fix.point.lat, fix.point.lon), int(self.radius_m))
        now = time.time()
        if key in self._cache:
            ts, best = self._cache[key]
            if now - ts < self.cache_ttl_s:
                return self._to_route(best, fix) if best else None

        best = self._query_best(fix.point.lat, fix.point.lon)
        self._cache[key] = (now, best)
        return self._to_route(best, fix) if best else None

    def _to_route(self, best: Optional[Tuple[str, float, float]], fix: GpsFix) -> Optional[RouteAdvisory]:
        if not best:
            return None
        name, lat, lon = best
        d = _haversine_m(fix.point.lat, fix.point.lon, lat, lon)
        speed = fix.speed_mps or self.assumed_speed_mps
        eta = d / max(speed, 2.0)
        return RouteAdvisory(
            destination_name=name,
            destination_point=GeoPoint(lat=lat, lon=lon),
            eta_sec=float(eta),
            distance_m=float(d),
            provider="osm_overpass",
            notes="Real-time POI lookup via OpenStreetMap Overpass; advisory-only.",
        )

    def _query_best(self, lat: float, lon: float) -> Optional[Tuple[str, float, float]]:
        # Query hospitals + emergency clinics.
        # Overpass QL: https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL
        q = f"""
[out:json][timeout:{int(self.timeout_s)}];
(
  node(around:{self.radius_m},{lat},{lon})[amenity=hospital];
  way(around:{self.radius_m},{lat},{lon})[amenity=hospital];
  relation(around:{self.radius_m},{lat},{lon})[amenity=hospital];
  node(around:{self.radius_m},{lat},{lon})[amenity=clinic][emergency=yes];
  way(around:{self.radius_m},{lat},{lon})[amenity=clinic][emergency=yes];
  relation(around:{self.radius_m},{lat},{lon})[amenity=clinic][emergency=yes];
);
out center tags;
""".strip()

        try:
            r = requests.post(self.overpass_url, data=q.encode("utf-8"), timeout=self.timeout_s)
            r.raise_for_status()
            data = r.json()
        except Exception:
            return None

        best: Optional[Tuple[float, str, float, float]] = None
        for el in data.get("elements", []):
            tags = el.get("tags", {}) or {}
            name = tags.get("name") or tags.get("operator") or "Hospital"
            if "lat" in el and "lon" in el:
                el_lat, el_lon = float(el["lat"]), float(el["lon"])
            else:
                center = el.get("center")
                if not center:
                    continue
                el_lat, el_lon = float(center.get("lat")), float(center.get("lon"))

            d = _haversine_m(lat, lon, el_lat, el_lon)
            if best is None or d < best[0]:
                best = (d, str(name), el_lat, el_lon)

        if best is None:
            return None
        _, name, bl_lat, bl_lon = best
        return (name, bl_lat, bl_lon)
