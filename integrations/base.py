"""Guardian Drive Integrations

This package defines *integration-ready interfaces* for real-world hooks:
- GPS: position/time source
- Telephony: emergency contact + dispatcher message
- Navigation: nearest ER recommendation / routing advisory
- Vehicle control: OEM-approved interface for limited vehicle actions

IMPORTANT SAFETY / LEGAL NOTE
-----------------------------
This repository intentionally ships *stubs and simulation providers*.
Real 911/PSAP integration and autonomous vehicle control require OEM
contracts, regulatory approval, and extensive safety validation.

We provide clean interfaces so you can plug in approved providers
(Tesla Fleet API, eCall, certified telematics) without rewriting the core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional, Dict, Any, List


@dataclass(frozen=True)
class GeoPoint:
    lat: float
    lon: float
    alt_m: Optional[float] = None


@dataclass(frozen=True)
class GpsFix:
    point: GeoPoint
    speed_mps: Optional[float] = None
    heading_deg: Optional[float] = None
    timestamp_unix: Optional[float] = None
    accuracy_m: Optional[float] = None


@dataclass(frozen=True)
class RouteAdvisory:
    destination_name: str
    destination_point: GeoPoint
    eta_sec: Optional[float]
    distance_m: Optional[float]
    provider: str
    notes: str = ""


@dataclass(frozen=True)
class DispatchMessage:
    title: str
    body: str
    meta: Dict[str, Any]


class GPSProvider(Protocol):
    def get_fix(self) -> Optional[GpsFix]:
        """Return the latest fix, or None if unavailable."""


class NavigationProvider(Protocol):
    def nearest_er(self, fix: GpsFix) -> Optional[RouteAdvisory]:
        """Return the best ER destination advisory for this fix."""


class TelephonyProvider(Protocol):
    def notify_emergency_contact(self, *, message: str, meta: Dict[str, Any]) -> None:
        """Notify the emergency contact(s). Implementation may send SMS/voice/app push."""

    def dispatch_simulation(self, *, message: DispatchMessage) -> None:
        """Simulation-only: print/log the dispatcher message."""


class VehicleControlProvider(Protocol):
    def request_safe_pull_over(self, *, reason: str, meta: Dict[str, Any]) -> None:
        """Request a safe pull-over using OEM-approved capabilities (if available).

        In this repo we ship NoOp/Simulation providers only.
        """


@dataclass
class IntegrationBundle:
    gps: GPSProvider
    nav: NavigationProvider
    tel: TelephonyProvider
    vehicle: VehicleControlProvider


