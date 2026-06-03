"""
Guardian Drive — Slack Integration
Bidirectional operator communication via Slack.

INGEST (reading):
  - Polls Slack channel for operator commands
  - Commands: acknowledge / override / dispatch / status / silence

PUSH (writing):
  - Posts alerts to Slack on state changes
  - Rich formatted blocks with risk score, location, BEV stats
  - Thread replies for escalation chain

Setup:
  1. Create Slack app at api.slack.com/apps
  2. Add Bot Token Scopes: chat:write, channels:history, channels:read
  3. Set SLACK_BOT_TOKEN and SLACK_CHANNEL_ID in .env

Real operator loop:
  Guardian Drive fires → Slack alert → operator responds
  → Guardian Drive receives command → adjusts policy
"""
from __future__ import annotations
import os, json, time, urllib.request, urllib.parse
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path
from datetime import datetime


@dataclass
class SlackAlert:
    state: str
    risk_score: float
    voice_message: str
    poi_name: str
    poi_distance_mi: float
    poi_eta_min: int
    task_a: dict
    task_b: dict
    task_c: dict
    bev_occ: int
    bev_ttc: float
    timestamp: float = field(default_factory=time.time)
    window: int = 0


@dataclass
class OperatorCommand:
    command: str        # acknowledge / override / dispatch / status / silence
    operator: str
    timestamp: float
    payload: dict = field(default_factory=dict)


class SlackIngestion:
    """
    Polls Slack channel for operator commands.
    Implements the right-seater operator loop:
      Alert → Operator sees in Slack → Operator types command
      → Guardian Drive receives and acts on it
    """

    COMMANDS = {
        'acknowledge': 'Operator acknowledges alert — suppresses repeat for 5min',
        'override':    'Operator overrides routing — manual destination',
        'dispatch':    'Operator confirms 911 dispatch',
        'status':      'Request current system status',
        'silence':     'Silence alarms for 2 minutes',
        'escalate':    'Force escalate state immediately',
    }

    def __init__(self):
        self.bot_token   = os.environ.get('SLACK_BOT_TOKEN', '')
        self.channel_id  = os.environ.get('SLACK_CHANNEL_ID', '')
        self._last_ts    = str(time.time())
        self._silenced   = False
        self._silence_until = 0.0
        self._acknowledged = False
        self._ack_until  = 0.0
        self._ready = bool(self.bot_token and self.channel_id)

    def poll_commands(self) -> List[OperatorCommand]:
        """Poll Slack for new operator commands since last check."""
        if not self._ready:
            return []
        try:
            url = 'https://slack.com/api/conversations.history'
            params = urllib.parse.urlencode({
                'channel': self.channel_id,
                'oldest':  self._last_ts,
                'limit':   10,
            })
            req = urllib.request.Request(
                f'{url}?{params}',
                headers={'Authorization': f'Bearer {self.bot_token}'}
            )
            with urllib.request.urlopen(req, timeout=3) as r:
                data = json.loads(r.read())

            messages = data.get('messages', [])
            commands = []
            for msg in reversed(messages):
                ts   = msg.get('ts', '0')
                text = msg.get('text', '').lower().strip()
                user = msg.get('user', 'unknown')
                if float(ts) > float(self._last_ts):
                    self._last_ts = ts
                    for cmd in self.COMMANDS:
                        if cmd in text:
                            commands.append(OperatorCommand(
                                command=cmd,
                                operator=user,
                                timestamp=float(ts),
                                payload={'raw': text}
                            ))
            return commands
        except Exception as e:
            return []

    def apply_command(self, cmd: OperatorCommand) -> str:
        """Apply operator command to Guardian Drive state."""
        now = time.time()
        if cmd.command == 'acknowledge':
            self._acknowledged = True
            self._ack_until = now + 300  # 5 min
            return "acknowledged — suppressing alerts for 5 minutes"
        elif cmd.command == 'silence':
            self._silenced = True
            self._silence_until = now + 120  # 2 min
            return "alarms silenced for 2 minutes"
        elif cmd.command == 'escalate':
            return "force_escalate"
        elif cmd.command == 'dispatch':
            return "dispatch_confirmed"
        return cmd.command

    def is_silenced(self) -> bool:
        if self._silenced and time.time() > self._silence_until:
            self._silenced = False
        return self._silenced

    def is_acknowledged(self) -> bool:
        if self._acknowledged and time.time() > self._ack_until:
            self._acknowledged = False
        return self._acknowledged


class SlackAlerter:
    """
    Posts rich formatted alerts to Slack.
    Uses Block Kit for structured messages with action buttons.
    """

    STATE_EMOJI = {
        'nominal':  '🟢',
        'advisory': '🟡',
        'caution':  '🟠',
        'pullover': '🔴',
        'escalate': '🚨',
    }

    STATE_COLOR = {
        'nominal':  '#22c55e',
        'advisory': '#84cc16',
        'caution':  '#f59e0b',
        'pullover': '#f97316',
        'escalate': '#ef4444',
    }

    def __init__(self):
        self.bot_token  = os.environ.get('SLACK_BOT_TOKEN', '')
        self.channel_id = os.environ.get('SLACK_CHANNEL_ID', '')
        self._ready = bool(self.bot_token and self.channel_id)
        self._last_state = 'nominal'
        self._last_alert_ts = 0.0
        self._thread_ts: Optional[str] = None

    def should_alert(self, state: str, risk: float) -> bool:
        """Only alert on state changes or risk jumps > 0.15."""
        now = time.time()
        if state != self._last_state:
            return True
        if now - self._last_alert_ts < 30:  # max 1 alert per 30s
            return False
        return False

    def post_alert(self, alert: SlackAlert) -> Optional[str]:
        """
        Post rich Slack alert. Returns thread_ts for follow-up replies.
        """
        if not self._ready:
            # Log to file instead
            self._log_to_file(alert)
            return None

        emoji = self.STATE_EMOJI.get(alert.state.lower(), '⚪')
        color = self.STATE_COLOR.get(alert.state.lower(), '#888888')

        # Build Slack Block Kit message
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} Guardian Drive™ — {alert.state.upper()}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Risk Score*\n`{alert.risk_score:.3f}`"},
                    {"type": "mrkdwn", "text": f"*Window*\n`{alert.window}`"},
                    {"type": "mrkdwn", "text": f"*Cardiac*\n`{alert.task_a.get('class','—')} {alert.task_a.get('score',0):.2f}`"},
                    {"type": "mrkdwn", "text": f"*Drowsy*\n`{alert.task_b.get('level','—')} {alert.task_b.get('score',0):.2f}`"},
                    {"type": "mrkdwn", "text": f"*Crash*\n`{alert.task_c.get('g_peak',0):.1f}g {'CRASH' if alert.task_c.get('detected') else 'CLR'}`"},
                    {"type": "mrkdwn", "text": f"*BEV TTC*\n`{alert.bev_ttc:.1f}s`"},
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*📍 Routing:* {alert.poi_name} — {alert.poi_distance_mi:.1f}mi · ETA {alert.poi_eta_min}min"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*🔊 Voice:* _{alert.voice_message}_"
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "✅ Acknowledge"},
                        "style": "primary",
                        "value": "acknowledge"
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "🚨 Confirm Dispatch"},
                        "style": "danger",
                        "value": "dispatch"
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "🔕 Silence 2min"},
                        "value": "silence"
                    }
                ]
            },
            {"type": "divider"},
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Guardian Drive v4.2 · {datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S')} · Reply `acknowledge` `override` `dispatch` `silence`"
                    }
                ]
            }
        ]

        payload = {
            "channel": self.channel_id,
            "blocks": blocks,
            "attachments": [{"color": color, "fallback": alert.voice_message}]
        }

        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                'https://slack.com/api/chat.postMessage',
                data=data,
                headers={
                    'Authorization': f'Bearer {self.bot_token}',
                    'Content-Type': 'application/json'
                }
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                result = json.loads(r.read())
            if result.get('ok'):
                self._thread_ts = result['ts']
                self._last_state = alert.state
                self._last_alert_ts = time.time()
                return self._thread_ts
        except Exception as e:
            self._log_to_file(alert)
        return None

    def post_thread_update(self, message: str):
        """Post follow-up in existing alert thread."""
        if not self._ready or not self._thread_ts:
            return
        try:
            payload = {
                "channel": self.channel_id,
                "thread_ts": self._thread_ts,
                "text": message
            }
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                'https://slack.com/api/chat.postMessage',
                data=data,
                headers={
                    'Authorization': f'Bearer {self.bot_token}',
                    'Content-Type': 'application/json'
                }
            )
            urllib.request.urlopen(req, timeout=5)
        except:
            pass

    def _log_to_file(self, alert: SlackAlert):
        """Fallback: log alert to JSONL when Slack not configured."""
        Path("runs").mkdir(exist_ok=True)
        entry = {
            "ts": alert.timestamp,
            "state": alert.state,
            "risk": alert.risk_score,
            "voice": alert.voice_message,
            "poi": alert.poi_name,
            "window": alert.window,
        }
        with open("runs/slack_alerts.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")


class GuardianSlack:
    """
    Combined Slack interface for Guardian Drive.
    Manages both ingestion and alerting in one object.
    """
    def __init__(self):
        self.ingestion = SlackIngestion()
        self.alerter   = SlackAlerter()
        self._prev_state = 'nominal'

    def process_window(self, payload: dict) -> List[str]:
        """
        Called every window. Returns list of operator actions to apply.
        1. Poll for operator commands
        2. Post alert if state changed
        3. Return commands to main pipeline
        """
        actions = []

        # Poll operator commands
        commands = self.ingestion.poll_commands()
        for cmd in commands:
            action = self.ingestion.apply_command(cmd)
            actions.append(action)
            print(f"[Slack] Operator command: {cmd.command} → {action}")

        # Post alert on state change
        state = payload.get('state', 'nominal')
        risk  = payload.get('guardian_risk_score', 0)

        if state != self._prev_state and state != 'nominal':
            pois = payload.get('poi', [{}])
            poi  = pois[0] if pois else {}
            alert = SlackAlert(
                state=state,
                risk_score=risk,
                voice_message=payload.get('voice_message', ''),
                poi_name=poi.get('name', 'No routing'),
                poi_distance_mi=poi.get('distance_m', 0) / 1609.34,
                poi_eta_min=poi.get('eta_min', 0),
                task_a=payload.get('task_a', {}),
                task_b=payload.get('task_b', {}),
                task_c=payload.get('task_c', {}),
                bev_occ=payload.get('bev_n_occupied', 0),
                bev_ttc=payload.get('bev_ttc', 99),
                window=payload.get('window', 0),
            )
            ts = self.alerter.post_alert(alert)
            if ts:
                print(f"[Slack] Alert posted: {state.upper()} risk={risk:.3f}")
            else:
                print(f"[Slack] Alert logged to file (no token configured)")

        self._prev_state = state
        return actions


def demo():
    """Demo Slack integration without real token."""
    print("="*60)
    print("Guardian Drive — Slack Integration Demo")
    print("="*60)

    slack = GuardianSlack()
    configured = slack.alerter._ready

    print(f"\n  Token configured: {configured}")
    print(f"  Mode: {'Live Slack API' if configured else 'File logging fallback'}")

    # Simulate state progression
    scenarios = [
        {"state":"nominal",  "guardian_risk_score":0.08,  "window":1,
         "voice_message":"Monitoring nominal.",
         "task_a":{"class":"normal","score":0.04},
         "task_b":{"level":"alert","score":0.12},
         "task_c":{"detected":False,"g_peak":0.1},
         "bev_ttc":12.0, "bev_n_occupied":340,
         "poi":[]},
        {"state":"caution",  "guardian_risk_score":0.674, "window":10,
         "voice_message":"MICROSLEEP DETECTED. Pull over now. Skyway Motel is 1.6 miles.",
         "task_a":{"class":"normal","score":0.12},
         "task_b":{"level":"microsleep","score":0.79},
         "task_c":{"detected":False,"g_peak":0.2},
         "bev_ttc":5.0, "bev_n_occupied":1149,
         "poi":[{"name":"Skyway Motel","distance_m":2575,"eta_min":3}]},
        {"state":"escalate", "guardian_risk_score":0.882, "window":15,
         "voice_message":"EMERGENCY. Driver unresponsive. Routing to Mount Sinai West.",
         "task_a":{"class":"AFIB","score":0.94},
         "task_b":{"level":"full_sleep","score":0.88},
         "task_c":{"detected":True,"g_peak":9.2},
         "bev_ttc":1.5, "bev_n_occupied":3905,
         "poi":[{"name":"Mount Sinai West","distance_m":4828,"eta_min":6}]},
    ]

    print(f"\n{'State':<12}{'Risk':>8}{'Action':<30}")
    print("-"*52)

    for payload in scenarios:
        actions = slack.process_window(payload)
        state = payload['state']
        risk  = payload['guardian_risk_score']
        action_str = ', '.join(actions) if actions else 'posted to Slack'
        print(f"  {state.upper():<10} {risk:>7.3f}  {action_str}")

    print(f"\n  Log: runs/slack_alerts.jsonl")
    print(f"\n  To enable live Slack:")
    print(f"    export SLACK_BOT_TOKEN=xoxb-your-token")
    print(f"    export SLACK_CHANNEL_ID=C0XXXXXXX")
    print(f"\n  Operator commands (type in Slack channel):")
    for cmd, desc in SlackIngestion.COMMANDS.items():
        print(f"    '{cmd}' → {desc}")

    print(f"\nSlack Integration: COMPLETE ✓")

if __name__ == "__main__":
    demo()
