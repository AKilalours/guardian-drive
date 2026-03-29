from __future__ import annotations

import json
import logging
import math
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "GuardianDrive/1.0 (rest-routing prototype)"


@dataclass
class POI:
    name: str
    category: str
    lat: float
    lon: float
    distance_km: float
    maps_url: str
    raw_tags: Dict[str, object]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _maps_url(lat: float, lon: float, label: str = "Destination") -> str:
    q = urllib.parse.quote(f"{lat},{lon} ({label})")
    return f"https://www.google.com/maps/search/?api=1&query={q}"


def _classify(tags: Dict[str, object]) -> str:
    if tags.get("highway") == "rest_area":
        return "rest_area"
    if tags.get("highway") == "services":
        return "highway_services"
    if tags.get("tourism") in {"motel", "hotel", "guest_house"}:
        return "lodging"
    if tags.get("amenity") == "cafe":
        return "coffee"
    if tags.get("amenity") == "fuel":
        return "fuel"
    if tags.get("amenity") == "parking":
        return "parking"
    return "other"


def query_rest_pois(lat: float, lon: float, *, radius_m: int = 12000, timeout_sec: int = 12) -> List[POI]:
    """
    Public OSM/Overpass lookup.
    Do not call this every frame. Cache it by rounded lat/lon or only refresh on meaningful movement.
    """
    q = f"""
    [out:json][timeout:{timeout_sec}];
    (
      node(around:{radius_m},{lat},{lon})[highway=rest_area];
      node(around:{radius_m},{lat},{lon})[highway=services];
      node(around:{radius_m},{lat},{lon})[tourism=motel];
      node(around:{radius_m},{lat},{lon})[tourism=hotel];
      node(around:{radius_m},{lat},{lon})[tourism=guest_house];
      node(around:{radius_m},{lat},{lon})[amenity=cafe];
      node(around:{radius_m},{lat},{lon})[amenity=fuel];
      node(around:{radius_m},{lat},{lon})[amenity=parking];
      way(around:{radius_m},{lat},{lon})[highway=rest_area];
      way(around:{radius_m},{lat},{lon})[highway=services];
      way(around:{radius_m},{lat},{lon})[tourism=motel];
      way(around:{radius_m},{lat},{lon})[tourism=hotel];
      way(around:{radius_m},{lat},{lon})[tourism=guest_house];
      way(around:{radius_m},{lat},{lon})[amenity=cafe];
      way(around:{radius_m},{lat},{lon})[amenity=fuel];
      way(around:{radius_m},{lat},{lon})[amenity=parking];
    );
    out center tags;
    """
    req = urllib.request.Request(
        OVERPASS_URL,
        data=q.encode("utf-8"),
        headers={"User-Agent": USER_AGENT, "Content-Type": "text/plain;charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec + 2) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    out: List[POI] = []
    for el in payload.get("elements", []):
        tags = el.get("tags", {}) or {}
        if "lat" in el and "lon" in el:
            plat, plon = float(el["lat"]), float(el["lon"])
        elif "center" in el:
            plat, plon = float(el["center"]["lat"]), float(el["center"]["lon"])
        else:
            continue
        category = _classify(tags)
        name = str(tags.get("name") or tags.get("brand") or category.replace("_", " ").title())
        dist = _haversine_km(lat, lon, plat, plon)
        out.append(
            POI(
                name=name,
                category=category,
                lat=plat,
                lon=plon,
                distance_km=round(dist, 2),
                maps_url=_maps_url(plat, plon, name),
                raw_tags=tags,
            )
        )
    out.sort(key=lambda x: x.distance_km)
    return out


class RestPOICache:
    def __init__(self, *, ttl_sec: float = 120.0, round_digits: int = 3) -> None:
        self.ttl_sec = ttl_sec
        self.round_digits = round_digits
        self._cache: Dict[str, Dict[str, object]] = {}

    def _key(self, lat: float, lon: float) -> str:
        return f"{round(lat, self.round_digits)}:{round(lon, self.round_digits)}"

    def nearby(self, lat: float, lon: float, *, radius_m: int = 12000) -> List[POI]:
        key = self._key(lat, lon)
        now = time.time()
        hit = self._cache.get(key)
        if hit and now - float(hit["ts"]) < self.ttl_sec:
            return hit["pois"]  # type: ignore[return-value]

        try:
            pois = query_rest_pois(lat, lon, radius_m=radius_m)
            self._cache[key] = {"ts": now, "pois": pois}
            return pois
        except Exception as e:
            logger.warning("rest_poi_osm query failed: %s", e)
            if hit:
                return hit["pois"]  # type: ignore[return-value]
            return []


def recommend_rest_stop(lat: float, lon: float, *, severity: str = "moderate", cache: Optional[RestPOICache] = None) -> Dict[str, object]:
    cache = cache or RestPOICache()
    pois = cache.nearby(lat, lon)

    if severity == "severe":
        preferred = {"rest_area": 0, "highway_services": 1, "lodging": 2, "parking": 3, "coffee": 4, "fuel": 5, "other": 9}
        recommendation = "Severe fatigue detected. Pull over at the nearest safe stop now. If you cannot recover quickly, do not resume driving."
        break_minutes = 30
    else:
        preferred = {"rest_area": 0, "coffee": 1, "highway_services": 2, "fuel": 3, "parking": 4, "lodging": 5, "other": 9}
        recommendation = "Moderate fatigue detected. Take a break soon. Aim for at least 15 minutes off the road."
        break_minutes = 15

    ranked = sorted(pois, key=lambda p: (preferred.get(p.category, 9), p.distance_km))
    top = ranked[:5]
    return {
        "kind": "rest_stop",
        "severity": severity,
        "recommended_break_minutes": break_minutes,
        "message": recommendation,
        "candidates": [
            {
                "name": p.name,
                "category": p.category,
                "lat": p.lat,
                "lon": p.lon,
                "distance_km": p.distance_km,
                "maps_url": p.maps_url,
            }
            for p in top
        ],
    }
