"""
integrations/gps_mac.py
Guardian Drive -- Real GPS from macOS / IP geolocation fallback

Run once:   python integrations/gps_mac.py
Continuous: python integrations/gps_mac.py --watch
"""
from __future__ import annotations
import argparse, json, subprocess, time, urllib.request


def _get_location():
    # IP geolocation fallback (works everywhere, city-level ~1-5km accuracy)
    try:
        # Try ip-api.com first (no key needed)
        with urllib.request.urlopen("http://ip-api.com/json/", timeout=6) as r:
            d = json.loads(r.read())
            if d.get("status") == "success":
                return {"lat": float(d["lat"]), "lon": float(d["lon"]),
                        "accuracy_m": 3000.0, "source": "ip_geolocation",
                        "city": d.get("city",""), "region": d.get("regionName","")}
    except Exception:
        pass
    return None


def push(loc, server="http://localhost:8000"):
    try:
        payload = json.dumps({"lat":loc["lat"],"lon":loc["lon"],
                              "accuracy_m":loc.get("accuracy_m",100.0),
                              "timestamp_unix":time.time()}).encode()
        req = urllib.request.Request(f"{server}/api/gps", data=payload,
                                     headers={"Content-Type":"application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=float, default=10.0)
    args = ap.parse_args()

    print(f"Guardian Drive GPS -> {args.server}")
    while True:
        loc = _get_location()
        if loc:
            print(f"  [{loc.get('source')}] {loc['lat']:.5f},{loc['lon']:.5f} {loc.get('city','')} +/-{loc['accuracy_m']:.0f}m")
            ok = push(loc, args.server)
            print(f"  -> {'OK' if ok else 'FAILED'}")
        else:
            print("  Could not get location")
        if not args.watch:
            break
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
