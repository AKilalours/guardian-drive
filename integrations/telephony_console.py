from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Any

from .base import TelephonyProvider, DispatchMessage


@dataclass
class ConsoleTelephony(TelephonyProvider):
    """Safe default: write notifications to stdout.

    Replace with Twilio/app push when you have approved channels.
    """

    def notify_emergency_contact(self, *, message: str, meta: Dict[str, Any]) -> None:
        print("\n[CONTACT NOTIFY]" + "-"*60)
        print(message)
        print("meta:")
        print(json.dumps(meta, indent=2))
        print("-"*75)

    def dispatch_simulation(self, *, message: DispatchMessage) -> None:
        print("\n[DISPATCH SIMULATION]" + "-"*56)
        print(message.title)
        print(message.body)
        print("meta:")
        print(json.dumps(message.meta, indent=2))
        print("-"*75)
