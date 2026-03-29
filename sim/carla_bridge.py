"""CARLA bridge (SIMULATION): route-to-hospital when Guardian Drive ESCALATE triggers.

This is the *credible* way to demo "autopilot drives to hospital":
- CARLA provides the vehicle + environment.
- Guardian Drive provides the safety state machine + dispatch.

IMPORTANT:
- This is SIMULATION ONLY. Do NOT claim OEM Autopilot/vehicle-control.

Run:
  # Terminal 1: CARLA server
  ./CarlaUE4.sh

  # Terminal 2: Guardian Drive server
  GD_PORT=8001 python -m server.app

  # Terminal 3: CARLA bridge
  pip install websockets
  GD_WS=ws://127.0.0.1:8001/ws python sim/carla_bridge.py
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Optional

import websockets


@dataclass
class CarlaRoutingState:
    active: bool = False
    last_level: str = ""
    destination_name: str = "Hospital (SIM)"


async def main() -> None:
    gd_ws = os.getenv("GD_WS", "ws://127.0.0.1:8001/ws")
    state = CarlaRoutingState()

    print(f"[CARLA] Connecting to Guardian Drive WS: {gd_ws}")
    async with websockets.connect(gd_ws) as ws:
        while True:
            msg = await ws.recv()
            p = json.loads(msg)
            pol = p.get("policy") or {}
            lvl = pol.get("level_name") or "NOMINAL"

            # Transition into ESCALATE triggers routing.
            if lvl == "ESCALATE" and state.last_level != "ESCALATE":
                route = p.get("route") or {}
                state.destination_name = route.get("destination_name") or "Hospital (SIM)"
                state.active = True
                print(f"[CARLA] ESCALATE received. Start route-to-hospital (SIM): {state.destination_name}")
                # TODO: connect to CARLA PythonAPI and set destination using an agent.

            if lvl != "ESCALATE" and state.active:
                state.active = False
                print("[CARLA] ESCALATE cleared. Stop routing (SIM).")

            state.last_level = lvl


if __name__ == "__main__":
    asyncio.run(main())
