from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class VoiceCommand:
    id: str
    text: str
    priority: int = 0
    category: str = "general"
    repeat: int = 1
    cooldown_sec: float = 6.0
    created_ts: float = field(default_factory=time.time)
    meta: Dict[str, object] = field(default_factory=dict)

    def as_payload(self) -> Dict[str, object]:
        return asdict(self)


class VoiceRuntime:
    """
    Backend-owned voice queue.

    Honest note:
    - This owns alert sequencing, dedupe, and priority.
    - In the current demo, the browser can still be the audio renderer.
    - Later you can swap in car audio or native TTS without changing the policy layer.
    """

    def __init__(self, *, max_history: int = 100) -> None:
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._history: Deque[VoiceCommand] = deque(maxlen=max_history)
        self._last_emit_by_key: Dict[str, float] = {}
        self._latest_for_frontend: Optional[VoiceCommand] = None
        self._latest_version: int = 0

    def enqueue(
        self,
        text: str,
        *,
        key: Optional[str] = None,
        priority: int = 0,
        category: str = "general",
        repeat: int = 1,
        cooldown_sec: float = 6.0,
        meta: Optional[Dict[str, object]] = None,
    ) -> Optional[VoiceCommand]:
        now = time.time()
        meta = meta or {}
        dedupe_key = key or f"{category}:{text.strip().lower()}"
        last = self._last_emit_by_key.get(dedupe_key, 0.0)
        if now - last < cooldown_sec:
            logger.debug("voice_runtime deduped key=%s", dedupe_key)
            return None
        self._last_emit_by_key[dedupe_key] = now

        cmd = VoiceCommand(
            id=str(uuid.uuid4()),
            text=text,
            priority=priority,
            category=category,
            repeat=max(1, int(repeat)),
            cooldown_sec=float(cooldown_sec),
            meta=meta,
        )
        # Lower numeric value = higher priority in PriorityQueue.
        self._queue.put_nowait((-priority, now, cmd))
        logger.warning("[VOICE_QUEUE] category=%s priority=%s text=%s meta=%s", category, priority, text, meta)
        return cmd

    async def drain_once(self) -> Optional[VoiceCommand]:
        if self._queue.empty():
            return None
        _prio, _ts, cmd = await self._queue.get()
        self._history.append(cmd)
        self._latest_for_frontend = cmd
        self._latest_version += 1
        logger.warning("[VOICE_EMIT] category=%s priority=%s text=%s", cmd.category, cmd.priority, cmd.text)
        return cmd

    async def pump_forever(self, *, tick_sec: float = 0.2) -> None:
        while True:
            try:
                await self.drain_once()
            except Exception:
                logger.exception("voice runtime pump failed")
            await asyncio.sleep(tick_sec)

    def latest_frontend_payload(self, *, after_version: int = -1) -> Optional[Dict[str, object]]:
        if self._latest_for_frontend is None:
            return None
        if after_version >= self._latest_version:
            return None
        payload = self._latest_for_frontend.as_payload()
        payload["version"] = self._latest_version
        payload["speak_in_browser"] = True
        return payload

    def recent_history(self, n: int = 10) -> List[Dict[str, object]]:
        return [c.as_payload() for c in list(self._history)[-n:]]


DEFAULT_MESSAGES = {
    "drowsy_advisory": "Stay alert. Consider a short break soon.",
    "drowsy_caution": "Fatigue detected. Please reduce speed and prepare to stop at a safe rest area.",
    "drowsy_pull_over": "Severe drowsiness detected. Pull over safely now.",
    "cardiac_urgent": "Medical risk detected. Pull over safely. Emergency guidance is being prepared.",
    "neuro_urgent": "Possible neurologic emergency detected. Pull over safely and seek emergency help now.",
    "crash_emergency": "Possible crash event detected. Emergency workflow active.",
}


def build_default_voice_runtime() -> VoiceRuntime:
    return VoiceRuntime()
