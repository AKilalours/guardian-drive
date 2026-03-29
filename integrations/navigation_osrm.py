import csv, math, time
from dataclasses import dataclass
from pathlib import Path
import requests

@dataclass
class NavAdvisory:
    destination_name: str
    lat: float
    lon: float
    eta_sec: float
    distance_m: float
    provider: str
    notes: str = ""

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2-lat1)
    dl = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

class OSRMHospitalNav:
    """
    Real routing using OSRM public backend.
    Failure modes:
      - timeout / rate limit / network down -> fallback to haversine ETA
    """
    def __init__(self, hospitals_csv: Path, osrm_base: str="http://router.project-osrm.org", timeout_s: float=2.5):
        self.osrm_base = osrm_base.rstrip("/")
        self.timeout_s = timeout_s
        self.hospitals = []
        with open(hospitals_csv, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                self.hospitals.append((row["name"], float(row["lat"]), float(row["lon"])))

    def nearest_er(self, fix):
        lat, lon = fix.point.lat, fix.point.lon

        # shortlist by haversine first to reduce OSRM calls
        ranked = sorted(self.hospitals, key=lambda h: haversine_m(lat, lon, h[1], h[2]))[:3]
        best = None

        for name, hlat, hlon in ranked:
            try:
                url = f"{self.osrm_base}/route/v1/driving/{lon},{lat};{hlon},{hlat}"
                r = requests.get(url, params={"overview":"false"}, timeout=self.timeout_s)
                r.raise_for_status()
                data = r.json()
                route = data["routes"][0]
                adv = NavAdvisory(
                    destination_name=name,
                    lat=hlat, lon=hlon,
                    eta_sec=float(route["duration"]),
                    distance_m=float(route["distance"]),
                    provider="osrm_public",
                    notes="Public OSRM; for production: self-host + caching + quotas."
                )
            except Exception:
                # fallback ETA: assume ~13.4 m/s (30 mph)
                d = haversine_m(lat, lon, hlat, hlon)
                adv = NavAdvisory(name, hlat, hlon, eta_sec=d/13.4, distance_m=d,
                                 provider="haversine_fallback",
                                 notes="OSRM failed -> fallback ETA")

            if best is None or adv.eta_sec < best.eta_sec:
                best = adv

            time.sleep(0.05)  # tiny spacing to be polite

        return best
