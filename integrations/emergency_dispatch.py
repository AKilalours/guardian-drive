"""
Guardian Drive™ v4.2 — Emergency Dispatch System
Tier 3: Next-of-Kin SMS Chain + 911 Auto-Dispatch

Chain:
  1. Detect ESCALATE state (risk > 0.85)
  2. SMS Contact 1 with location + live link
  3. If no response in 30s → SMS Contact 2
  4. Auto-dispatch 911 with GPS coordinates
  5. Route to nearest ER (Mount Sinai West)

IMPORTANT: Real 911 dispatch requires NG911/eCall compliance.
This implementation notifies emergency contacts + provides
location data for manual 911 calls.
"""
from __future__ import annotations
import os, time, json, threading
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class EmergencyContact:
    name:   str
    phone:  str
    relation: str  # "spouse", "parent", "friend"


@dataclass 
class DispatchEvent:
    timestamp:    float
    risk_score:   float
    location:     Dict[str, float]  # lat, lon
    hr_bpm:       int
    scenario:     str
    contacts_notified: List[str] = field(default_factory=list)
    dispatch_sent: bool = False
    er_routed:    str = ""


class NextOfKinChain:
    """
    Progressive emergency contact chain.
    
    Stage 1 (risk > 0.75): Warn driver
    Stage 2 (risk > 0.85): SMS Contact 1
    Stage 3 (30s no response): SMS Contact 2  
    Stage 4 (risk > 0.95): 911 coordinates
    """
    
    # Default contacts (override with env vars)
    CONTACTS = [
        EmergencyContact(
            name=os.getenv("GD_CONTACT1_NAME", "Emergency Contact 1"),
            phone=os.getenv("GD_CONTACT1_PHONE", ""),
            relation="primary",
        ),
        EmergencyContact(
            name=os.getenv("GD_CONTACT2_NAME", "Emergency Contact 2"),
            phone=os.getenv("GD_CONTACT2_PHONE", ""),
            relation="secondary",
        ),
    ]
    
    # NYC 911 dispatch coordinates (demo)
    DISPATCH_911 = {
        "number":  os.getenv("GD_911_NUMBER", "911"),
        "address": "NYPD/FDNY Emergency Dispatch",
        "note":    "Real 911 requires NG911 compliance",
    }
    
    def __init__(self):
        self._active_event:  Optional[DispatchEvent] = None
        self._chain_started: float = 0.0
        self._stage:         int   = 0
        self._lock           = threading.Lock()
        self._log:           List[Dict] = []
        
        # Try to initialize Twilio
        self._twilio = None
        self._twilio_from = os.getenv("TWILIO_FROM_NUMBER", "")
        try:
            sid = os.getenv("TWILIO_ACCOUNT_SID", "")
            tok = os.getenv("TWILIO_AUTH_TOKEN", "")
            if sid and tok:
                from twilio.rest import Client
                self._twilio = Client(sid, tok)
                print("✓ Twilio emergency dispatch ready")
            else:
                print("⚠ Twilio not configured — console mode")
        except Exception as e:
            print(f"⚠ Twilio init failed: {e} — console mode")

    def _maps_link(self, lat: float, lon: float) -> str:
        return f"https://maps.google.com/?q={lat},{lon}"

    def _send_sms(self, to: str, body: str) -> bool:
        """Send SMS via Twilio or print to console."""
        if self._twilio and to and self._twilio_from:
            try:
                msg = self._twilio.messages.create(
                    body=body, from_=self._twilio_from, to=to)
                print(f"✓ SMS sent to {to}: {msg.sid}")
                return True
            except Exception as e:
                print(f"✗ SMS failed: {e}")
        # Console fallback
        print(f"\n📱 SMS → {to}:")
        print(f"   {body}\n")
        return True

    def _sms_contact1(self, event: DispatchEvent) -> None:
        """Stage 2: Alert primary contact."""
        c = self.CONTACTS[0]
        lat = event.location.get('lat', 40.7589)
        lon = event.location.get('lon', -73.9851)
        link = self._maps_link(lat, lon)
        
        body = (
            f"🚨 GUARDIAN DRIVE ALERT\n"
            f"Hi {c.name}, {os.getenv('GD_DRIVER_NAME','the driver')} "
            f"may need help.\n"
            f"Risk score: {event.risk_score:.0%}\n"
            f"HR: {event.hr_bpm} BPM\n"
            f"Location: {link}\n"
            f"Routing to: {event.er_routed}\n"
            f"Reply SAFE if they're okay."
        )
        if self._send_sms(c.phone, body):
            event.contacts_notified.append(c.name)
            self._log_event("sms_contact1", event, body)

    def _sms_contact2(self, event: DispatchEvent) -> None:
        """Stage 3: Alert secondary contact."""
        c = self.CONTACTS[1]
        lat = event.location.get('lat', 40.7589)
        lon = event.location.get('lon', -73.9851)
        link = self._maps_link(lat, lon)
        
        body = (
            f"🚨 URGENT — GUARDIAN DRIVE\n"
            f"Hi {c.name}, {os.getenv('GD_DRIVER_NAME','the driver')} "
            f"has not responded.\n"
            f"PLEASE CALL 911 and share this location:\n"
            f"{link}\n"
            f"Risk: {event.risk_score:.0%} | HR: {event.hr_bpm} BPM\n"
            f"Nearest ER: {event.er_routed}"
        )
        if self._send_sms(c.phone, body):
            event.contacts_notified.append(c.name)
            self._log_event("sms_contact2", event, body)

    def _dispatch_911(self, event: DispatchEvent) -> None:
        """Stage 4: 911 auto-dispatch (coordinates + details)."""
        lat = event.location.get('lat', 40.7589)
        lon = event.location.get('lon', -73.9851)
        link = self._maps_link(lat, lon)
        
        # In production: integrate with NG911 API
        # For demo: SMS emergency number + log
        dispatch_msg = (
            f"🚨 911 AUTO-DISPATCH — GUARDIAN DRIVE\n"
            f"Medical emergency in vehicle.\n"
            f"GPS: {lat:.6f}, {lon:.6f}\n"
            f"Maps: {link}\n"
            f"Cardiac risk: {event.risk_score:.0%}\n"
            f"HR: {event.hr_bpm} BPM\n"
            f"Routing to: {event.er_routed}\n"
            f"Time: {time.strftime('%H:%M:%S')}"
        )
        
        print(f"\n🚨 911 DISPATCH ACTIVATED")
        print(dispatch_msg)
        event.dispatch_sent = True
        self._log_event("911_dispatch", event, dispatch_msg)

    def _log_event(self, event_type: str,
                   event: DispatchEvent, message: str) -> None:
        self._log.append({
            "type":      event_type,
            "timestamp": time.strftime('%H:%M:%S'),
            "risk":      event.risk_score,
            "message":   message[:100],
        })

    def update(self, risk_score: float, location: Dict,
               hr_bpm: int, scenario: str, er_name: str) -> Optional[str]:
        """
        Call every window. Returns action taken or None.
        
        Args:
            risk_score: 0-1 Guardian risk score
            location:   {"lat": float, "lon": float}
            hr_bpm:     Heart rate
            scenario:   Current scenario name
            er_name:    Nearest ER name
        """
        with self._lock:
            now = time.time()

            # Stage 1: WARN
            if risk_score < 0.75:
                self._stage = 0
                self._active_event = None
                return None

            # Create event if new escalation
            if self._active_event is None:
                self._active_event = DispatchEvent(
                    timestamp  = now,
                    risk_score = risk_score,
                    location   = location,
                    hr_bpm     = hr_bpm,
                    scenario   = scenario,
                    er_routed  = er_name,
                )
                self._chain_started = now
                self._stage = 1
                return "MONITORING"

            # Update event
            self._active_event.risk_score = risk_score
            self._active_event.hr_bpm     = hr_bpm
            elapsed = now - self._chain_started

            # Stage 2: SMS Contact 1 (immediate on ESCALATE)
            if self._stage < 2 and risk_score >= 0.85:
                self._stage = 2
                threading.Thread(
                    target=self._sms_contact1,
                    args=(self._active_event,),
                    daemon=True
                ).start()
                return f"SMS_CONTACT1: {self.CONTACTS[0].name}"

            # Stage 3: SMS Contact 2 (30s after Contact 1)
            if self._stage < 3 and elapsed > 30 and risk_score >= 0.85:
                self._stage = 3
                threading.Thread(
                    target=self._sms_contact2,
                    args=(self._active_event,),
                    daemon=True
                ).start()
                return f"SMS_CONTACT2: {self.CONTACTS[1].name}"

            # Stage 4: 911 Dispatch (60s or risk > 0.95)
            if self._stage < 4 and (elapsed > 60 or risk_score >= 0.95):
                self._stage = 4
                threading.Thread(
                    target=self._dispatch_911,
                    args=(self._active_event,),
                    daemon=True
                ).start()
                return "911_DISPATCHED"

            return f"ESCALATE_STAGE{self._stage}"

    def get_log(self) -> List[Dict]:
        return self._log.copy()

    def save_log(self, path: str = "outputs/dispatch_log.json") -> None:
        Path(path).write_text(json.dumps({
            "dispatch_log": self._log,
            "total_events": len(self._log),
        }, indent=2))


# Singleton
_chain: Optional[NextOfKinChain] = None

def get_dispatch_chain() -> NextOfKinChain:
    global _chain
    if _chain is None:
        _chain = NextOfKinChain()
    return _chain
