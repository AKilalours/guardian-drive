"""
sim/carla_bridge.py
Guardian Drive — ESCALATE → CARLA route-to-hospital  [SIMULATION ONLY]
Adapted from: carla-simulator/scenario_runner RouteScenario pattern

Called by policy/state_machine.py when ESCALATE fires.
Gracefully degrades to dry-run if CARLA is not installed.

Run demo:
    python sim/carla_bridge.py --demo
"""
from __future__ import annotations
import argparse, json, logging, math, os, time
from dataclasses import dataclass

log = logging.getLogger("guardian_drive.sim")


@dataclass
class RouteResult:
    success:        bool
    distance_m:     float
    eta_s:          float
    message:        str
    is_simulation:  bool = True   # ALWAYS true — we never control a real vehicle


class CarlaBridge:
    """
    Listens for ESCALATE events → triggers CARLA autonomous route.
    Falls back to a logged dry-run when CARLA is not reachable.

    Usage (from policy/state_machine.py):
        bridge = CarlaBridge()
        result = bridge.on_escalate(
            hospital_lat=37.78, hospital_lon=-122.41,
            hospital_name="UCSF Medical Center",
        )
    """

    def __init__(self, host: str = "localhost", port: int = 2000, timeout: float = 8.0):
        self.host, self.port, self.timeout = host, port, timeout
        self._carla_ok = False
        self._client = self._world = None
        self._try_connect()

    def _try_connect(self) -> None:
        try:
            import carla
            self._client = carla.Client(self.host, self.port)
            self._client.set_timeout(self.timeout)
            self._world  = self._client.get_world()
            self._carla_ok = True
            log.info(f"CARLA connected at {self.host}:{self.port}")
        except ImportError:
            log.warning("carla not installed — dry-run mode.  Install: pip install carla==0.9.15")
        except Exception as e:
            log.warning(f"CARLA unreachable ({e}) — dry-run mode")

    # ── public ───────────────────────────────────────────────────────────────

    def on_escalate(
        self,
        hospital_lat:  float,
        hospital_lon:  float,
        hospital_name: str   = "Nearest ER",
        reason:        str   = "ESCALATE",
        patient_state: str   = "unknown",
    ) -> RouteResult:
        """Call this when ESCALATE triggers in policy/state_machine.py"""
        log.info(f"[SIM] ESCALATE → {hospital_name} ({hospital_lat:.4f},{hospital_lon:.4f})")
        dist = self._haversine(hospital_lat, hospital_lon)
        eta  = dist / 8.0   # ~30 km/h urban speed

        if self._carla_ok:
            return self._carla_route(hospital_lat, hospital_lon, hospital_name, dist, eta)
        return self._dry_run(hospital_name, hospital_lat, hospital_lon, reason, dist, eta)

    # ── internal ─────────────────────────────────────────────────────────────

    def _carla_route(self, lat, lon, name, dist, eta) -> RouteResult:
        try:
            import carla
            actors = self._world.get_actors()
            ego    = next((a for a in actors if "vehicle" in a.type_id), None)
            if ego is None:
                bp  = self._world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
                ego = self._world.spawn_actor(bp, self._world.get_map().get_spawn_points()[0])
            ego.set_autopilot(True)
            log.info(f"[CARLA SIM] Autopilot engaged → {name}  {dist:.0f}m  ETA {eta:.0f}s")
            self._log_event(lat, lon, name, dist, eta)
            return RouteResult(True, round(dist,1), round(eta,1),
                               f"[CARLA SIM] Routing to {name}", True)
        except Exception as e:
            log.error(f"CARLA route failed: {e}")
            return self._dry_run(name, lat, lon, "carla_error", dist, eta)

    def _dry_run(self, name, lat, lon, reason, dist, eta) -> RouteResult:
        msg = (f"[DRY-RUN SIM] Would route to {name} ({lat:.4f},{lon:.4f}) "
               f"~{dist:.0f}m / ETA {eta:.0f}s — CARLA not connected")
        log.info(msg)
        self._log_event(lat, lon, name, dist, eta)
        return RouteResult(True, round(dist,1), round(eta,1), msg, True)

    def _haversine(self, lat2: float, lon2: float) -> float:
        """Distance from a fixed origin (replace with actual GPS in integrations/gps.py)"""
        lat1, lon1 = 37.7749, -122.4194   # SF placeholder
        R = 6_371_000
        φ1,φ2 = math.radians(lat1), math.radians(lat2)
        dφ,dλ = math.radians(lat2-lat1), math.radians(lon2-lon1)
        a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
        return R * 2 * math.asin(math.sqrt(a))

    def _log_event(self, lat, lon, name, dist, eta) -> None:
        os.makedirs("runs", exist_ok=True)
        entry = {"event":"CARLA_ROUTE","ts":time.time(),
                 "hospital":name,"lat":lat,"lon":lon,
                 "distance_m":round(dist,1),"eta_s":round(eta,1),"simulation":True}
        with open("runs/sim_events.jsonl","a") as f:
            f.write(json.dumps(entry)+"\n")


# ── standalone demo ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--lat",  type=float, default=37.7833)
    ap.add_argument("--lon",  type=float, default=-122.4167)
    ap.add_argument("--name", default="SF General Hospital")
    args = ap.parse_args()

    print("\n[SIMULATION DEMO]  Guardian Drive — CARLA Bridge")
    print("NOTE: No real vehicle is controlled.\n")
    bridge = CarlaBridge(args.host, args.port)

    if args.demo:
        r = bridge.on_escalate(args.lat, args.lon, args.name,
                                reason="arrhythmia_demo", patient_state="high_risk")
        print(f"\nResult:")
        print(f"  success    = {r.success}")
        print(f"  distance   = {r.distance_m} m")
        print(f"  ETA        = {r.eta_s} s")
        print(f"  simulation = {r.is_simulation}")
        print(f"  message    = {r.message}")
