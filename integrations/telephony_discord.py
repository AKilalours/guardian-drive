"""
Discord webhook "telephony" (free).

Env:
  DISCORD_WEBHOOK_URL=<your webhook>

Optional:
  GD_DISCORD_WEBHOOK_URL=<same>  # accepted as fallback

This is NOT voice calling; it is an escalation notification channel.
"""
from __future__ import annotations

import os, requests
from integrations.base import DispatchMessage


class DiscordWebhookTelephony:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url.strip()

    @classmethod
    def from_env(cls) -> "DiscordWebhookTelephony":
        url = (os.getenv("DISCORD_WEBHOOK_URL") or os.getenv("GD_DISCORD_WEBHOOK_URL") or "").strip()
        if not url:
            raise RuntimeError("Missing DISCORD_WEBHOOK_URL (or GD_DISCORD_WEBHOOK_URL)")
        return cls(url)

    def dispatch_simulation(self, message: DispatchMessage) -> None:
        # Keep payload small + readable
        meta = message.meta or {}
        content = f"**{message.title}**\n{message.body}\n\n```json\n{_safe_json(meta)}\n```"
        r = requests.post(self.webhook_url, json={"content": content}, timeout=6)
        r.raise_for_status()


def _safe_json(obj) -> str:
    import json
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return json.dumps({"meta": str(obj)}, indent=2, ensure_ascii=False)
