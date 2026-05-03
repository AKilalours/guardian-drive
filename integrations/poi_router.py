"""
integrations/poi_router.py
Guardian Drive -- Enhanced POI Router by Impairment Type

Routes to the right type of POI based on impairment state:

SLEEPY    -> Starbucks / Costa Coffee / coffee shop (nearest, open now)
DROWSY    -> Rest area / service station / parking
FATIGUED  -> Motel / hotel (driver needs sleep, not just coffee)
ESCALATE  -> Hospital emergency room

All queries use OpenStreetMap Overpass API -- free, no API key.

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
"""
import httpx
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class POIResult:
    name:      str
    lat:       float
    lon:       float
    distance_m: float
    poi_type:  str
    maps_url:  str
    eta_min:   Optional[int] = None

def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2-lat1)
    dl = math.radians(lon2-lon1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# OSM Overpass queries per impairment type
QUERIES = {
    "cafe": """
[out:json][timeout:10];
(
  node["amenity"="cafe"](around:{r},{lat},{lon});
  node["amenity"="coffee_shop"](around:{r},{lat},{lon});
  node["name"~"Starbucks|Costa|Dunkin|Tim Hortons",i](around:{r},{lat},{lon});
);
out 5;
""",
    "rest_area": """
[out:json][timeout:10];
(
  node["highway"="rest_area"](around:{r},{lat},{lon});
  node["amenity"="parking"](around:{r},{lat},{lon});
  node["amenity"="fuel"](around:{r},{lat},{lon});
);
out 5;
""",
    "motel": """
[out:json][timeout:10];
(
  node["tourism"="motel"](around:{r},{lat},{lon});
  node["tourism"="hotel"](around:{r},{lat},{lon});
  node["tourism"="hostel"](around:{r},{lat},{lon});
);
out 5;
""",
    "hospital": """
[out:json][timeout:10];
(
  node["amenity"="hospital"](around:{r},{lat},{lon});
  node["amenity"="clinic"](around:{r},{lat},{lon});
  node["emergency"="yes"](around:{r},{lat},{lon});
);
out 5;
""",
}

def find_poi(lat: float, lon: float,
             poi_type: str,
             radius_m: int = 3000) -> Optional[POIResult]:
    """
    Find nearest POI of given type using OSM Overpass API.
    poi_type: "cafe" | "rest_area" | "motel" | "hospital"
    """
    query_template = QUERIES.get(poi_type, QUERIES["cafe"])
    query = query_template.format(r=radius_m, lat=lat, lon=lon)

    try:
        resp = httpx.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            timeout=12.0
        )
        data = resp.json()
        elements = data.get("elements", [])
        if not elements:
            return None

        # Find nearest
        best = None
        best_dist = float("inf")
        for el in elements:
            elat = el.get("lat", 0)
            elon = el.get("lon", 0)
            dist = _haversine(lat, lon, elat, elon)
            if dist < best_dist:
                best_dist = dist
                best = el

        if not best:
            return None

        name     = (best.get("tags", {}).get("name") or
                    poi_type.replace("_", " ").title())
        elat     = best["lat"]
        elon     = best["lon"]
        maps_url = (f"https://www.google.com/maps/dir/"
                    f"{lat},{lon}/{elat},{elon}")
        eta_min  = int(best_dist / 500)  # rough 30kph estimate

        return POIResult(
            name       = name,
            lat        = elat,
            lon        = elon,
            distance_m = round(best_dist),
            poi_type   = poi_type,
            maps_url   = maps_url,
            eta_min    = eta_min,
        )

    except Exception as e:
        print(f"[POIRouter] Query failed: {e}")
        return None

def get_poi_for_impairment(lat: float, lon: float,
                            impairment: str) -> Optional[POIResult]:
    """
    Get the right POI type for the detected impairment.
    """
    mapping = {
        "sleepy":     "cafe",
        "drowsy":     "rest_area",
        "fatigued":   "motel",
        "microsleep": "hospital",
        "escalate":   "hospital",
    }
    poi_type = mapping.get(impairment.lower(), "cafe")
    return find_poi(lat, lon, poi_type)

if __name__ == "__main__":
    # Test from Brooklyn
    lat, lon = 40.5948, -73.9715
    for state in ["sleepy", "drowsy", "fatigued", "microsleep"]:
        print(f"\nSearching for {state} POI...")
        result = get_poi_for_impairment(lat, lon, state)
        if result:
            print(f"  Found: {result.name}")
            print(f"  Distance: {result.distance_m}m (~{result.eta_min}min)")
            print(f"  Maps: {result.maps_url}")
        else:
            print(f"  No {state} POI found within 3km")
