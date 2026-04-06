"""
Run from guardian-drive root:
    python patch_voice.py
Adds voice alerts to server/app.py pipeline loop.
"""
from pathlib import Path

p = Path("server/app.py")
txt = p.read_text()

VOICE_FN = '''
# ── Voice alerts (macOS) ───────────────────────────────────────────────────────
import subprocess as _sp, threading as _thr

_VOICE_ENABLED = os.getenv("GD_VOICE","1").strip().lower() in {"1","true","yes","on"}
_LAST_VOICE_LV: str = ""
_LAST_VOICE_TS: float = 0.0
_VOICE_SCRIPTS = {
    "ADVISORY": "Heads up. Early fatigue signs detected. Consider a short break soon.",
    "CAUTION":  "Warning. Fatigue confirmed. Please find a rest stop within the next few miles.",
    "PULLOVER": "Alert. Severe drowsiness. Pull over now and rest immediately.",
    "ESCALATE": "Emergency. Medical event detected. Routing to nearest hospital. Emergency contacts notified.",
}
def _speak(msg:str)->None:
    if not _VOICE_ENABLED: return
    def _r():
        try: _sp.run(["say","-v","Samantha","-r","155",msg],timeout=12,capture_output=True)
        except: pass
    _thr.Thread(target=_r,daemon=True).start()
'''

VOICE_CALL = '''
        # Voice alert on level change
        global _LAST_VOICE_LV,_LAST_VOICE_TS
        _now_v=time.time()
        _vscript=_VOICE_SCRIPTS.get(lv,"")
        _cooldown=45.0 if lv!="ESCALATE" else 60.0
        if _vscript and (lv!=_LAST_VOICE_LV or (_now_v-_LAST_VOICE_TS)>_cooldown):
            # Append POI name if available
            _poi_r=payload.get("poi") or {}
            if _poi_r.get("name") and lv in ("ADVISORY","CAUTION","PULLOVER"):
                _vscript+=f" There is a {_poi_r['name']} in {_poi_r.get('distance_mi','?')} miles."
            elif payload.get("route",{}) and lv=="ESCALATE":
                _vscript+=f" Routing to {payload.get('route',{}).get('destination_name','')}."
            _speak(_vscript)
            _LAST_VOICE_LV=lv; _LAST_VOICE_TS=_now_v
'''

if "_speak" not in txt:
    txt = txt.replace("async def _pipeline():", VOICE_FN + "\nasync def _pipeline():")
    print("✓ Added voice functions")

if "_VOICE_SCRIPTS" in txt and "_LAST_VOICE_LV" not in txt:
    # already has fn, add call
    pass

if "# Voice alert on level change" not in txt:
    # Insert after broadcast
    txt = txt.replace(
        "        await _broadcast(payload)\n\n        if lv==\"ESCALATE\":",
        "        await _broadcast(payload)\n" + VOICE_CALL + "\n        if lv==\"ESCALATE\":"
    )
    if "# Voice alert on level change" not in txt:
        txt = txt.replace(
            "        await _broadcast(payload)\n        if lv==\"ESCALATE\":",
            "        await _broadcast(payload)\n" + VOICE_CALL + "\n        if lv==\"ESCALATE\":"
        )
    print("✓ Added voice call in pipeline")

p.write_text(txt)
print("Done — restart server to activate voice alerts")
print("Test: say -v Samantha -r 155 'Guardian Drive voice online'")
