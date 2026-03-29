# integrations/discord_webhook.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any

import requests


@dataclass
class DiscordNotifier:
    webhook_url: str

    @staticmethod
    def from_env() -> Optional["DiscordNotifier"]:
        url = (os.getenv("GD_DISCORD_WEBHOOK_URL") or os.getenv("DISCORD_WEBHOOK_URL") or "").strip()
        if not url:
            return None
        return DiscordNotifier(webhook_url=url)

    def send(self, title: str, body: str, meta: Optional[Dict[str, Any]] = None) -> bool:
        lines = [f"**{title}**", body]
        if meta:
            # keep it short; Discord has limits
            meta_txt = json.dumps(meta, ensure_ascii=False)[:1500]
            lines.append(f"```json\n{meta_txt}\n```")
        payload = {"content": "\n".join(lines)}
        try:
            r = requests.post(self.webhook_url, json=payload, timeout=6)
            return r.status_code in (200, 204)
        except Exception:
            return False
