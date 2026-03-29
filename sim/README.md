# CARLA Simulation Bridge

This folder exists for one reason: a **credible** demo of "route-to-hospital" without making unsafe or false OEM claims.

## What is real vs simulation

- **Real:** Guardian Drive policy state machine, dispatch packet generation, GPS ingestion, hospital lookup.
- **Simulation:** vehicle control (autopilot), route execution, environment.

## Minimal demo (listening mode)

The included `carla_bridge.py` listens to Guardian Drive WebSocket and prints when **ESCALATE** triggers.

```bash
pip install websockets
GD_WS=ws://127.0.0.1:8001/ws python sim/carla_bridge.py
```

## Full autopilot routing (optional)

To actually drive a vehicle in CARLA you must:
1) Install CARLA and its PythonAPI
2) Spawn an ego vehicle
3) Use a navigation agent (e.g., BehaviorAgent) and set a destination

That integration is intentionally separate because CARLA setup differs by OS and version.
