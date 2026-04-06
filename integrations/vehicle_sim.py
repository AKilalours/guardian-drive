"""
integrations/vehicle_sim.py
Guardian Drive — Vehicle control stubs

NoOpVehicleControl: no-op stub (default)
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class VehicleSnapshot:
    ts:           float = field(default_factory=time.time)
    speed_kph:    float = 0.0
    throttle:     float = 0.0
    brake:        float = 0.0
    steering:     float = 0.0
    gear:         str   = "D"
    engine_on:    bool  = True
    source:       str   = "noop"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts":        self.ts,
            "speed_kph": self.speed_kph,
            "throttle":  self.throttle,
            "brake":     self.brake,
            "steering":  self.steering,
            "gear":      self.gear,
            "engine_on": self.engine_on,
            "source":    self.source,
        }


class NoOpVehicleControl:
    """
    No-op vehicle control stub.
    All commands are accepted and silently ignored.
    snapshot() returns a zeroed VehicleSnapshot.
    """

    def snapshot(self) -> Dict[str, Any]:
        return VehicleSnapshot(source="noop").to_dict()

    def set_throttle(self, v: float) -> None:   pass
    def set_brake(self, v: float) -> None:       pass
    def set_steering(self, v: float) -> None:    pass
    def set_hazard_lights(self, on: bool) -> None: pass
    def set_horn(self, on: bool) -> None:         pass
    def close(self) -> None:                      pass

    def __repr__(self) -> str:
        return "NoOpVehicleControl()"
