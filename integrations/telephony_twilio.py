"""Twilio telephony provider (OPTIONAL).

This provider can send SMS or place outbound calls to *pre-configured numbers*.

IMPORTANT:
- This is NOT a certified 911/PSAP integration.
- Calling emergency services requires a compliant eCall/NG911 pathway.
- Use this to notify an emergency contact, or to run safe demos.

Env vars:
  TWILIO_ACCOUNT_SID
  TWILIO_AUTH_TOKEN
  TWILIO_FROM_NUMBER
  GD_EMERGENCY_CONTACT_NUMBER

Install:
  pip install twilio
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any

from .base import TelephonyProvider, DispatchMessage


@dataclass
class TwilioTelephony(TelephonyProvider):
    to_number: str

    def __post_init__(self):
        try:
            from twilio.rest import Client
        except Exception as e:
            raise RuntimeError("twilio not installed. pip install twilio")

        sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        tok = os.getenv("TWILIO_AUTH_TOKEN", "")
        frm = os.getenv("TWILIO_FROM_NUMBER", "")
        if not (sid and tok and frm):
            raise RuntimeError("Missing TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN/TWILIO_FROM_NUMBER")

        self._Client = Client
        self._client = Client(sid, tok)
        self._from = frm

    def notify_emergency_contact(self, *, message: str, meta: Dict[str, Any]) -> None:
        body = message
        self._client.messages.create(to=self.to_number, from_=self._from, body=body)

    def dispatch_simulation(self, *, message: DispatchMessage) -> None:
        # For Twilio, treat simulation as SMS to the emergency contact.
        self.notify_emergency_contact(message=f"{message.title}\n{message.body}", meta=message.meta)
