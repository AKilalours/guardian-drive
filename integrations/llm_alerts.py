"""
integrations/llm_alerts.py
Guardian Drive -- LLM-Powered Alert Explanation System

Uses GPT-4o to generate natural language explanations of:
- Why the safety state machine escalated
- What the physiological signals mean clinically
- What the driver should do next
- Risk factors from AV context

This demonstrates LLM integration for safety-critical AI systems --
a key capability in Tesla FSD v12+ which uses LLMs for
scene understanding and driver communication.

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
"""

import os
import json
import time
import threading
from pathlib import Path
from typing import Optional

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

SYSTEM_PROMPT = """You are Guardian Drive's safety AI assistant.
You analyze multimodal physiological and AV perception data to explain
driver impairment risk in clear, calm, actionable language.

Rules:
- Be concise (2-3 sentences max per explanation)
- Never cause panic -- calm and factual tone
- Always suggest a specific action
- Never claim to be a medical device
- Always include disclaimer: "This is a research prototype, not medical advice"

You understand these signals:
- PERCLOS: % eye closure (>25% = sleepy, >80% = microsleep)
- RMSSD: HRV metric (< 20ms = fatigue, < 30ms = early fatigue)
- EAR: Eye Aspect Ratio (< 0.18 = eyes closing)
- TCN score: Low-arousal classifier output (0-1)
- SQI: Signal quality (< 0.30 = abstain)
- AV context: Traffic density, intersection proximity, speed
"""

def explain_alert(
    state:         str,
    risk_score:    float,
    impairment:    str,
    perclos:       float,
    hrv_rmssd:     float,
    tcn_prob:      float,
    sqi:           float,
    n_objects:     int,
    near_intersect:bool,
    speed_kph:     float,
    poi_name:      Optional[str] = None,
    blocking:      bool = True,
) -> dict:
    """
    Generate LLM explanation of current safety state.
    Returns dict with explanation, recommendation, and confidence.
    """
    if not OPENAI_API_KEY:
        return _fallback_explanation(state, impairment, poi_name)

    user_msg = f"""
Current Guardian Drive safety alert:

STATE: {state}
RISK SCORE: {risk_score:.3f}
IMPAIRMENT TYPE: {impairment.upper()}

Physiological signals:
- TCN low-arousal probability: {tcn_prob:.3f}
- PERCLOS (30s): {perclos:.2%}
- HRV RMSSD: {hrv_rmssd:.1f} ms
- Signal quality (SQI): {sqi:.3f}

AV context:
- Nearby objects: {n_objects}
- Near intersection: {near_intersect}
- Speed: {speed_kph:.0f} kph

{'Nearest safe stop: ' + poi_name if poi_name else ''}

Generate:
1. EXPLANATION: Why is the system alerting? (1-2 sentences, calm tone)
2. RECOMMENDATION: What should the driver do right now? (1 sentence, specific)
3. DISCLAIMER: One line safety disclaimer

Format as JSON: {{"explanation": "...", "recommendation": "...", "disclaimer": "..."}}
"""

    try:
        import urllib.request
        payload = json.dumps({
            "model":       "gpt-4o",
            "max_tokens":  300,
            "temperature": 0.3,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg}
            ]
        }).encode()

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data    = payload,
            headers = {
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            }
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data    = json.loads(resp.read())
            content = data["choices"][0]["message"]["content"]

        # Parse JSON response
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result  = json.loads(content.strip())
        result["model"]   = "gpt-4o"
        result["state"]   = state
        result["latency"] = data.get("usage", {})
        return result

    except Exception as e:
        print(f"[LLM] GPT-4o call failed: {e}")
        return _fallback_explanation(state, impairment, poi_name)

def _fallback_explanation(state: str, impairment: str,
                           poi_name: Optional[str]) -> dict:
    """Rule-based fallback when API unavailable."""
    explanations = {
        "NOMINAL":   "All physiological signals are within normal range.",
        "ADVISORY":  "Early fatigue indicators detected. Your reaction time may be slightly reduced.",
        "CAUTION":   "Moderate impairment signals detected across multiple channels.",
        "PULLOVER":  "Significant impairment detected. Continuing to drive is not recommended.",
        "ESCALATE":  "Critical physiological event detected. Immediate action required.",
    }
    recommendations = {
        "sleepy":     f"Pull over and get a coffee.{' ' + poi_name + ' is nearby.' if poi_name else ''}",
        "drowsy":     f"Take a 15-minute break at the next rest stop.{' ' + poi_name + ' is nearby.' if poi_name else ''}",
        "fatigued":   f"Stop driving. You need sleep, not coffee.{' ' + poi_name + ' is nearby.' if poi_name else ''}",
        "microsleep": "Pull over immediately. You experienced a microsleep episode.",
        "alert":      "Continue driving safely.",
    }
    return {
        "explanation":     explanations.get(state, "Safety alert triggered."),
        "recommendation":  recommendations.get(impairment, "Take a break."),
        "disclaimer":      "Research prototype only. Not medical advice.",
        "model":           "rule-based fallback",
        "state":           state,
    }

def speak_explanation(explanation: dict,
                       voice: str = "Samantha") -> None:
    """Speak LLM explanation via TTS."""
    import subprocess
    msg = (f"{explanation.get('explanation','')} "
           f"{explanation.get('recommendation','')}")
    threading.Thread(
        target=lambda: subprocess.run(
            ["say", "-v", voice, "-r", "140", msg],
            capture_output=True, timeout=20),
        daemon=True
    ).start()

def log_explanation(explanation: dict,
                     log_dir: str = "data/llm_logs") -> None:
    """Log LLM explanation to JSONL."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "explanations.jsonl"
    entry = {**explanation, "timestamp": time.time()}
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    import sys
    key = os.getenv("OPENAI_API_KEY","")
    print(f"OpenAI API key: {'set' if key else 'NOT SET -- set OPENAI_API_KEY'}")

    result = explain_alert(
        state          = "CAUTION",
        risk_score     = 0.58,
        impairment     = "drowsy",
        perclos        = 0.22,
        hrv_rmssd      = 28.0,
        tcn_prob       = 0.65,
        sqi            = 0.82,
        n_objects      = 8,
        near_intersect = True,
        speed_kph      = 65.0,
        poi_name       = "Starbucks - 0.4 miles",
    )
    print(json.dumps(result, indent=2))
