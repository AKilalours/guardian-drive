from dataclasses import dataclass
from typing import Dict, Any

# If your project already defines DispatchMessage, use that import instead:
# from integrations.base import DispatchMessage
@dataclass
class DispatchMessage:
    title: str
    body: str
    meta: Dict[str, Any]

from integrations.discord_telephony import DiscordTelephony  # <-- change to your real file path

telephony = DiscordTelephony.from_env()

msg = DispatchMessage(
    title="EMERGENCY: Demo Alert",
    body="Driver fatigue detected. Sending notification.",
    meta={"lat": 40.7357, "lon": -74.1724, "vehicle_id": "GD-001"}
)

telephony.dispatch_simulation(msg)
print("sent ✅")
