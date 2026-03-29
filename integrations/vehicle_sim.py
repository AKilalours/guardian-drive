from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from .base import VehicleControlProvider


@dataclass
class NoOpVehicleControl(VehicleControlProvider):
    """Default provider: does nothing.

    Real vehicle control (autopilot, steering/braking) is OEM-only and
    should not be attempted via reverse engineering.

    Provide an OEM-approved implementation (e.g., telematics that can request
    hazard lights, reduce speed, or perform a safe pull-over in a constrained ODD).
    """

    def request_safe_pull_over(self, *, reason: str, meta: Dict[str, Any]) -> None:
        print(f"[VEHICLE CONTROL] NoOp pull-over request. reason={reason} meta_keys={list(meta.keys())}")
