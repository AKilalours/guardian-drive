from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional

router = APIRouter(prefix="/api/sensor", tags=["sensor"])

SEAT_NODE = None


def bind_seat_node(node) -> None:
    global SEAT_NODE
    SEAT_NODE = node


class SeatPacket(BaseModel):
    t0: float
    fs_hz: float = 250.0
    samples: List[float] = Field(default_factory=list)
    contact_ok: bool = False
    motion_score: float = 0.0
    seq: Optional[int] = None


@router.post("/seat_ecg")
def post_seat_ecg(packet: SeatPacket):
    if SEAT_NODE is None:
        return {"ok": False, "error": "seat node not bound"}
    SEAT_NODE.ingest_packet(packet.model_dump())
    return {"ok": True, "n": len(packet.samples)}


@router.get("/seat_ecg/status")
def get_seat_ecg_status():
    if SEAT_NODE is None:
        return {"ok": False, "error": "seat node not bound"}
    return {"ok": True, "status": SEAT_NODE.snapshot()}
