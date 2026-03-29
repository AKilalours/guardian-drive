from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

try:
    import obd
except Exception:
    obd = None

@dataclass
class OBDVehicleTelemetry:
    port: Optional[str] = None
    connection: Any = None
    last_ok: bool = False

    def __post_init__(self):
        if obd is None:
            raise RuntimeError("python-obd not installed. pip install obd")
        self.connection = obd.OBD(self.port)  # auto-connect if port None

    def snapshot(self) -> Dict[str, Any]:
        # read-only: safe + credible “real” integration
        if not self.connection or not self.connection.is_connected():
            return {"provider":"obd", "connected": False}

        def q(cmd):
            r = self.connection.query(cmd)
            return None if r.is_null() else str(r.value)

        data = {
            "provider":"obd",
            "connected": True,
            "speed": q(obd.commands.SPEED),
            "rpm": q(obd.commands.RPM),
            "throttle": q(obd.commands.THROTTLE_POS),
            "engine_load": q(obd.commands.ENGINE_LOAD),
        }
        self.last_ok = True
        return data

    def close(self):
        try:
            if self.connection:
                self.connection.close()
        except Exception:
            pass
