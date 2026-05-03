"""
integrations/emergency_contact.py
Guardian Drive -- Emergency Contact Notification

On ESCALATE or MICROSLEEP:
1. Sends SMS to pre-configured emergency contact with:
   - Driver name
   - Current GPS location (Google Maps link)
   - Impairment type and severity
   - Nearest hospital name and distance
   - Timestamp
2. Opens macOS Messages app as fallback
3. Logs dispatch attempt to telemetry

Uses Twilio API (free trial: $15 credit, no card needed for trial)
or macOS Messages as fallback.

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
"""
import os
import time
import subprocess
import threading
from pathlib import Path

EMERGENCY_CONTACT_NAME  = os.getenv("GD_EMERGENCY_CONTACT_NAME", "Emergency Contact")
EMERGENCY_CONTACT_PHONE = os.getenv("GD_EMERGENCY_CONTACT_PHONE", "")
DRIVER_NAME             = os.getenv("GD_DRIVER_NAME", "Guardian Drive User")
TWILIO_SID              = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN            = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM             = os.getenv("TWILIO_FROM_NUMBER", "")

def _build_message(impairment: str, lat: float, lon: float,
                   hospital: str, severity: float) -> str:
    maps_url = f"https://maps.google.com/?q={lat},{lon}"
    return (
        f"GUARDIAN DRIVE ALERT\n"
        f"Driver: {DRIVER_NAME}\n"
        f"Status: {impairment.upper()} detected (severity {severity:.0%})\n"
        f"Location: {maps_url}\n"
        f"Nearest hospital: {hospital}\n"
        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"This is an automated safety alert."
    )

def send_sms(impairment: str, lat: float, lon: float,
             hospital: str, severity: float,
             blocking: bool = False) -> dict:
    """
    Send emergency SMS to configured contact.
    Returns status dict for telemetry logging.
    """
    msg = _build_message(impairment, lat, lon, hospital, severity)

    def _send():
        result = {"method": None, "success": False,
                  "timestamp": time.time(), "message_preview": msg[:80]}

        # Method 1: Twilio API
        if TWILIO_SID and TWILIO_TOKEN and EMERGENCY_CONTACT_PHONE:
            try:
                from twilio.rest import Client
                client = Client(TWILIO_SID, TWILIO_TOKEN)
                message = client.messages.create(
                    body=msg,
                    from_=TWILIO_FROM,
                    to=EMERGENCY_CONTACT_PHONE
                )
                result.update({"method": "twilio", "success": True,
                               "sid": message.sid})
                print(f"[EmergencyContact] SMS sent via Twilio: {message.sid}")
                return result
            except Exception as e:
                print(f"[EmergencyContact] Twilio failed: {e}")

        # Method 2: macOS Messages (opens app -- user sends manually)
        try:
            phone = EMERGENCY_CONTACT_PHONE or "your emergency contact"
            script = f'''
tell application "Messages"
    activate
    set targetBuddy to "{phone}"
    set targetService to id of 1st account whose service type = iMessage
    set textMessage to "{msg.replace(chr(10), " | ")}"
    send textMessage to buddy targetBuddy of account id targetService
end tell
'''
            subprocess.run(["osascript", "-e", script],
                           capture_output=True, timeout=10)
            result.update({"method": "macos_messages", "success": True})
            print("[EmergencyContact] macOS Messages opened")
        except Exception as e:
            print(f"[EmergencyContact] macOS Messages failed: {e}")
            result.update({"method": "failed", "success": False})

        return result

    if blocking:
        return _send()
    else:
        threading.Thread(target=_send, daemon=True).start()
        return {"method": "async", "success": None}

def log_dispatch(result: dict, log_dir: str = "data/emergency_logs"):
    """Log emergency dispatch attempt to JSONL."""
    import json
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "dispatch_log.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    print("Emergency Contact System Test")
    print(f"Contact: {EMERGENCY_CONTACT_NAME} ({EMERGENCY_CONTACT_PHONE or 'not configured'})")
    print("Set GD_EMERGENCY_CONTACT_PHONE in .env to enable SMS")
    result = send_sms(
        impairment="microsleep",
        lat=40.5948, lon=-73.9715,
        hospital="Jersey City Medical Center",
        severity=0.95, blocking=True)
    print(f"Result: {result}")
