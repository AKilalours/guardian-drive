from __future__ import annotations

"""
Guardian Drive(tm) v4.3 -- Complete Real-Time Server
WebSocket + SSE + full pipeline + auto-escalation + Discord + POI

Run:
    GD_LIVE_ONLY=0 GD_ENABLE_WEBCAM=1 python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import math
import os
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Optional, Set

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse, JSONResponse, HTMLResponse,
    StreamingResponse, Response
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from acquisition.simulator import GuardianSimulator, SCENARIO_PARAMS
from acquisition.seat_ecg_node import SeatECGNode
from sqi.compute import compute_sqi
from features.extract import extract_features
from policy.fusion import FusionEngine
from policy.state_machine import SafetyStateMachine
from acquisition.models import (
    AlertLevel, ArrhythmiaClass, CrashSeverity,
    PolicyAction, RiskState, SensorFrame, TaskLabel, FS_ECG,
)
from server.routes_sensor import router as sensor_router, bind_seat_node
from integrations.gps_runtime import RuntimeGPS
from integrations.navigation_local import LocalHospitalNav
from integrations.navigation_osm import OSMHospitalNav
from integrations.telephony_console import ConsoleTelephony
from integrations.vehicle_sim import NoOpVehicleControl
from integrations.base import DispatchMessage
from integrations.discord_webhook import DiscordNotifier
# nuScenes real BEV
try:
    from acquisition.nuscenes_bev import NuScenesBEV as _NuScenesBEV
    _NUSCENES = _NuScenesBEV()
    if _NUSCENES.available:
        print(f"[GD] nuScenes BEV loaded — real perception data active")
    else:
        _NUSCENES = None
except Exception as _e:
    print(f"[GD] nuScenes not available: {_e}")
    _NUSCENES = None


try:
    from integrations.telephony_twilio import TwilioTelephony
except Exception:
    TwilioTelephony = None

try:
    from integrations.vision_webcam import WebcamMonitor
    _WEBCAM_CLS = WebcamMonitor
except Exception:
    _WEBCAM_CLS = None

# env
def _eb(k,d): return os.getenv(k,"1" if d else "0").strip().lower() in {"1","true","yes","on"}
def _ef(k,d): return float(os.getenv(k,str(d)))
def _ei(k,d): return int(os.getenv(k,str(d)))

GD_LIVE_ONLY    = _eb("GD_LIVE_ONLY", False)
GD_WEBCAM       = _eb("GD_ENABLE_WEBCAM", False)
GD_WEBCAM_IDX   = _ei("GD_WEBCAM_INDEX", 0)
GD_SEAT_ECG     = _eb("GD_ENABLE_SEAT_ECG", True)
GD_SEAT_FS      = _ef("GD_SEAT_ECG_FS_HZ", 250.0)
GD_SEAT_STALE   = _ef("GD_SEAT_ECG_STALE_SEC", 3.0)
GD_STEP         = _ef("GD_STEP_SEC", 4.0)
GD_WINDOW       = _ef("GD_WINDOW_SEC", 30.0)
GD_NAV_REFRESH  = _ef("GD_NAV_REFRESH_SEC", 15.0)
GD_NAV_MOVE     = _ef("GD_NAV_MOVE_M", 75.0)
GD_DISCORD      = os.getenv("GD_DISCORD_WEBHOOK_URL","")
GD_SCENARIO     = os.getenv("GD_SCENARIO","normal")

BASE_DIR   = Path(__file__).resolve().parent.parent
SERVER_DIR = Path(__file__).resolve().parent
STATIC_DIR = SERVER_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

_POI_MAP = {
    "NOMINAL":None,"ADVISORY":["cafe","restaurant","fast_food"],
    "CAUTION":["cafe","motel","hotel","rest_area"],
    "PULLOVER":["parking","rest_area","motel","hotel"],
    "ESCALATE":["hospital","clinic","doctors"],"INACTIVE":None,
}
_POI_LABEL = {
    "ADVISORY":"Rest stop nearby -- take a break soon",
    "CAUTION":"Find a rest stop -- fatigue confirmed",
    "PULLOVER":"Pull over now -- severe drowsiness",
    "ESCALATE":"Nearest emergency care",
}
_SUMMARY = {
    "NOMINAL":"All systems nominal. Drive safely.",
    "ADVISORY":"Early fatigue signs. Consider a short break soon.",
    "CAUTION":"Fatigue confirmed. Find a rest stop within a few miles.",
    "PULLOVER":"Severe drowsiness. Pull over immediately.",
    "ESCALATE":"Medical emergency. Routing to nearest emergency care.",
    "INACTIVE":"System inactive -- waiting for sensor data.",
}

_clients: Set[WebSocket] = set()
_state: dict = {}
_scenario: str = GD_SCENARIO
_lock = asyncio.Lock()
_CAM = None
_SEAT_NODE = SeatECGNode(max_seconds=max(120.0,GD_WINDOW*4.0), fs_hz=GD_SEAT_FS)
_LAST_DISC: float = 0.0
_RUNTIME: dict = {"startup":time.time(),"fusion":None,"source":None}
_NAV_CACHE: dict = {"fix":None,"route":None,"ts":0.0}


class ScenarioReq(BaseModel):
    scenario: str

class GpsReq(BaseModel):
    lat: float; lon: float
    accuracy_m: Optional[float]=None; timestamp_unix: Optional[float]=None


def _g(obj,name,default=None):
    if obj is None: return default
    if isinstance(obj,dict): return obj.get(name,default)
    return getattr(obj,name,default)

def _fix_d(fix):
    if not fix: return None
    return {"lat":fix.point.lat,"lon":fix.point.lon,"accuracy_m":fix.accuracy_m,
            "maps":f"https://www.google.com/maps?q={fix.point.lat:.6f},{fix.point.lon:.6f}",
            "timestamp_unix":fix.timestamp_unix}

def _route_d(route):
    if not route: return None
    return {"destination_name":route.destination_name,"lat":route.destination_point.lat,
            "lon":route.destination_point.lon,"eta_sec":route.eta_sec,
            "distance_m":route.distance_m,"provider":route.provider}

def _hav(a1,o1,a2,o2):
    R=6371000; x=math.radians(o2-o1)*math.cos(math.radians((a1+a2)/2)); y=math.radians(a2-a1)
    return math.sqrt(x*x+y*y)*R

def _level(action):
    try:
        lvl=getattr(action,"level",None) or getattr(action,"state","NOMINAL")
        if hasattr(lvl,"name"): return lvl.name
        return str(lvl).upper().split(".")[-1]
    except: return "NOMINAL"

def _reason(action):
    for a in ["log_reason","display_message","summary","voice_message"]:
        v=getattr(action,a,None)
        if v: return str(v)[:120]
    r=getattr(action,"reasons",[])
    return " | ".join(str(x) for x in r)[:120] if r else "Monitoring."

def _rest(rs):
    d=getattr(rs,"drowsiness",None)
    if not d or d.abstained: return None
    s=float(d.score)
    if s>=0.85: return {"recommended_stop_min":30,"urgency":"critical","message":"Severe drowsiness confirmed. Pull over now and rest at least 30 minutes."}
    if s>=0.65: return {"recommended_stop_min":20,"urgency":"high","message":"Significant fatigue. Plan a rest stop within the next few minutes."}
    if s>=0.45: return {"recommended_stop_min":10,"urgency":"moderate","message":"Early fatigue signs. A short break is recommended soon."}
    return None

def _ds(arr,n=500):
    if arr is None: return None
    x=np.asarray(arr,dtype=np.float32).flatten()
    if x.size<2: return None
    return x.tolist() if x.size<=n else x[np.linspace(0,x.size-1,n).astype(np.int32)].tolist()

def _ta(rs):
    a=rs.arrhythmia
    if not a: return {"score":0.0,"cls":"unknown","abstained":True,"hr_bpm":0,"confidence":0.0}
    return {"score":round(float(_g(a,"confidence",0) or 0),3),"confidence":round(float(_g(a,"confidence",0) or 0),3),
            "cls":str(getattr(getattr(a,"cls",None),"value","unknown")),
            "abstained":bool(_g(a,"abstained",False)),"hr_bpm":round(float(_g(a,"hr_bpm",0) or 0),1),
            "rr_irr":round(float(_g(a,"rr_irr",0) or 0),4),"p_frac":round(float(_g(a,"p_frac",0) or 0),3),
            "reason":str(_g(a,"reason","") or "")[:80],"features_used":list(getattr(a,"features_used",[]) or [])}

def _tb(rs,wm):
    d=rs.drowsiness
    base={"score":round(float(_g(d,"score",0) or 0),3) if d else 0.0,
          "confidence":round(float(_g(d,"confidence",0) or 0),3) if d else 0.0,
          "abstained":bool(_g(d,"abstained",False)) if d else True,
          "reason":str(_g(d,"reason","") or "")[:80] if d else "",
          "hr_contrib":round(float(_g(d,"hr_contrib",0) or 0),3) if d else 0.0,
          "hrv_contrib":round(float(_g(d,"hrv_contrib",0) or 0),3) if d else 0.0,
          "resp_contrib":round(float(_g(d,"resp_contrib",0) or 0),3) if d else 0.0,
          "eda_contrib":round(float(_g(d,"eda_contrib",0) or 0),3) if d else 0.0}
    if wm:
        base.update({"ear":wm.get("ear"),"perclos":wm.get("perclos_30s"),
                     "blink_rate":wm.get("blink_rate_30s"),"yawns_30s":wm.get("yawn_events_30s"),
                     "eyes_closed":wm.get("eyes_closed"),"eyes":wm.get("eyes"),
                     "drowsy_cam":wm.get("drowsy_score"),"face_detected":wm.get("face_detected",False)})
    return base

def _tc(rs):
    c=rs.crash
    if not c: return {"score":0.0,"detected":False,"severity":"none","g_peak":0.0}
    sev=getattr(c,"severity",None); sv=int(getattr(sev,"value",0) or 0)
    conf=float(_g(c,"confidence",0) or 0)
    det=bool(_g(c,"detected",False))
    score=max(conf,0.95) if det and sv>=2 else (max(conf,0.55) if det else 0.0)
    return {"score":round(score,3),"detected":det,"severity":str(getattr(sev,"name","none")).lower() if sev else "none",
            "g_peak":round(float(_g(c,"g_peak",0) or 0),2),"confidence":round(conf,3),
            "reason":str(_g(c,"reason","") or "")[:80]}

def _feats(fb):
    try:
        return {"hr_bpm":float(fb.ecg.hr_bpm or 0),"hrv_rmssd":float(fb.ecg.hrv_rmssd or 0),
                "hrv_sdnn":float(fb.ecg.hrv_sdnn or 0),"rr_irregularity":float(fb.ecg.rr_irregularity or 0),
                "p_wave_fraction":float(fb.ecg.p_wave_fraction or 0),
                "resp_rate_bpm":float(fb.resp.rate_bpm or 0) if fb.resp.rate_bpm else None,
                "eda_scl_mean":float(fb.eda.scl_mean or 0) if fb.eda.scl_mean else None,
                "posture_score":float(fb.imu.posture_score or 0),"crash_g_peak":float(fb.imu.crash_g_peak or 0),
                "ecg_source":str(getattr(fb.ecg,"source","") or ""),
                "seat_ecg_quality":float(getattr(fb.sqi,"seat_ecg_quality",0) or 0)}
    except: return {}


async def _poi(lat,lon,amenities,radius=2500):
    try:
        import httpx
        q="".join(f'node["amenity"="{a}"](around:{radius},{lat},{lon});' for a in amenities)
        async with httpx.AsyncClient(timeout=8.0) as c:
            r=await c.post("https://overpass-api.de/api/interpreter",data={"data":f"[out:json][timeout:8];({q});out body 5;"})
            els=r.json().get("elements",[])
            if not els: return None
            n=min(els,key=lambda e:(e.get("lat",lat)-lat)**2+(e.get("lon",lon)-lon)**2)
            t=n.get("tags",{}); elat,elon=n.get("lat",lat),n.get("lon",lon)
            dist=_hav(lat,lon,elat,elon)
            return {"name":t.get("name",t.get("amenity",amenities[0]).title()),"lat":elat,"lon":elon,
                    "distance_m":round(dist),"distance_mi":round(dist/1609.34,2),
                    "amenity":t.get("amenity",amenities[0]),"maps_url":f"https://www.google.com/maps?q={elat:.6f},{elon:.6f}"}
    except: return None


def _nav(fix,nav_osm,nav_local,level="NOMINAL"):
    # Only route to hospital during ESCALATE
    if not fix: return None
    if level not in ("ESCALATE","PULLOVER"): return None
    now=time.monotonic(); cf=_NAV_CACHE["fix"]; cr=_NAV_CACHE["route"]; ct=_NAV_CACHE["ts"]
    need=cr is None or (now-ct)>=GD_NAV_REFRESH
    if not need and cf:
        moved=_hav(float(cf.point.lat),float(cf.point.lon),float(fix.point.lat),float(fix.point.lon))
        if moved>=GD_NAV_MOVE: need=True
    if need:
        route=nav_osm.nearest_er(fix) or nav_local.nearest_er(fix)
        if route: _NAV_CACHE.update({"fix":fix,"route":route,"ts":now})
    return _NAV_CACHE["route"]


async def _discord(state:dict):
    if not GD_DISCORD: return
    try:
        import httpx
        lv=state.get("level","NOMINAL")
        col=0xFF0000 if lv=="ESCALATE" else 0xFF8C00 if lv in ("PULLOVER","CAUTION") else 0xFFD600
        ta,tb,tc=state.get("task_a",{}),state.get("task_b",{}),state.get("task_c",{})
        gps=state.get("gps",{}); poi=state.get("poi"); rest=state.get("rest_guidance")
        fields=[
            {"name":"Alert Level","value":f"**{lv}**","inline":True},
            {"name":"Reason","value":state.get("reason","--")[:200],"inline":False},
            {"name":"Task A ECG","value":f"cls={ta.get('cls','--')} conf={ta.get('confidence',0):.2f} HR={ta.get('hr_bpm',0):.0f}bpm","inline":True},
            {"name":"Task B Drowsy","value":f"score={tb.get('score',0):.2f}","inline":True},
            {"name":"Task C Crash","value":f"{'DETECTED' if tc.get('detected') else 'none'} g={tc.get('g_peak',0):.1f}","inline":True},
        ]
        if gps.get("lat"): fields.append({"name":"GPS","value":f"[{gps['lat']:.5f},{gps['lon']:.5f}]({gps.get('maps','#')})","inline":True})
        if rest: fields.append({"name":"Rest","value":rest.get("message","--"),"inline":False})
        if poi and poi.get("name"):
            em="🏥" if lv=="ESCALATE" else "☕"
            fields.append({"name":f"{em} Nearest","value":f"**{poi['name']}** -- {poi.get('distance_mi','?')} mi\n[Maps]({poi.get('maps_url','#')})","inline":False})
        async with httpx.AsyncClient(timeout=5.0) as c:
            await c.post(GD_DISCORD,json={"username":"Guardian Drive(tm)","embeds":[{
                "title":f"{'ESCALATE' if lv=='ESCALATE' else ''} GUARDIAN DRIVE -- {lv}",
                "color":col,"fields":fields,
                "timestamp":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                "footer":{"text":"Guardian Drive(tm) Safety System"}}]})
    except: pass


async def _broadcast(payload:dict):
    if not _clients: return
    msg=json.dumps(payload,separators=(",",":"),ensure_ascii=False)
    dead=set()
    for ws in list(_clients):
        try: await ws.send_text(msg)
        except: dead.add(ws)
    _clients.difference_update(dead)


@dataclass
class CompatAction:
    timestamp:float=field(default_factory=time.monotonic)
    level:AlertLevel=AlertLevel.NOMINAL
    display_message:str=""; voice_message:str=""; log_reason:str=""
    escalate_911:bool=False; hospital_advisory:bool=False
    persistence_sec:float=0.0; corroborated_by:list=field(default_factory=list)
    def to_dict(self):
        return {"timestamp":float(self.timestamp),"level":int(self.level.value),"level_name":self.level.name,
                "display_message":self.display_message,"voice_message":self.voice_message,
                "log_reason":self.log_reason,"escalate_911":bool(self.escalate_911),
                "hospital_advisory":bool(self.hospital_advisory),
                "persistence_sec":float(self.persistence_sec),"corroborated_by":list(self.corroborated_by)}

def _norm(raw)->CompatAction:
    if raw is None: return CompatAction(level=AlertLevel.INACTIVE,display_message="No policy output.")
    if hasattr(raw,"level") and hasattr(raw,"display_message"): return raw
    sn=_g(raw,"state","nominal"); sm=str(_g(raw,"summary","") or "Monitoring.")
    rs=list(_g(raw,"reasons",[]) or []); rk=str(_g(raw,"route_kind","none") or "none")
    lmap={"nominal":AlertLevel.NOMINAL,"advisory":AlertLevel.ADVISORY,"caution":AlertLevel.CAUTION,
          "pull_over":AlertLevel.PULLOVER,"escalate":AlertLevel.ESCALATE}
    lv=lmap.get(sn,AlertLevel.NOMINAL)
    pd=bool(_g(raw,"prepare_dispatch",False)); nc=bool(_g(raw,"notify_contact",False))
    return CompatAction(timestamp=float(_g(raw,"ts",time.monotonic()) or time.monotonic()),
        level=lv,display_message=sm,voice_message=sm,
        log_reason=" | ".join(str(r) for r in rs) if rs else sm,
        escalate_911=bool(lv==AlertLevel.ESCALATE or pd or nc),
        hospital_advisory=bool(rk=="er"),corroborated_by=[str(r) for r in rs])

def _tb2(task,kind):
    if task is None: return {"score":0.0,"details":{}}
    det=dict(getattr(task,"details",{}) or {}); score=0.0
    if kind=="task_a":
        ab=bool(getattr(task,"abstained",False)); cls=getattr(task,"cls",None)
        cv=getattr(cls,"value",str(cls) if cls else "unknown")
        abn=(not ab) and cv not in {"normal","noisy","unknown"}
        score=float(getattr(task,"confidence",0) or 0) if abn else 0.0
        det.update({"cls":cv,"hr_bpm":_g(task,"hr_bpm"),"rr_irr":float(_g(task,"rr_irr",0) or 0),"p_frac":float(_g(task,"p_frac",0) or 0)})
    elif kind=="task_b":
        score=float(getattr(task,"score",0) or 0)
        det.update({"hr_contrib":float(getattr(task,"hr_contrib",0) or 0),"hrv_contrib":float(getattr(task,"hrv_contrib",0) or 0),
                    "resp_contrib":float(getattr(task,"resp_contrib",0) or 0),"eda_contrib":float(getattr(task,"eda_contrib",0) or 0)})
    elif kind=="task_c":
        det2=bool(getattr(task,"detected",False)); sev=getattr(task,"severity",None)
        sv=int(getattr(sev,"value",0) or 0); conf=float(getattr(task,"confidence",0) or 0)
        score=max(conf,0.95) if det2 and sv>=2 else (max(conf,0.55) if det2 else 0.0)
        det.update({"detected":det2,"severity":sv,"g_peak":float(getattr(task,"g_peak",0) or 0)})
    return {"score":float(score),"details":det}

def _rpi(rs):
    return {"task_a":_tb2(rs.arrhythmia,"task_a"),"task_b":_tb2(rs.drowsiness,"task_b"),
            "task_c":_tb2(rs.crash,"task_c"),"task_d":{"score":0.0,"details":{}}}

def _rpol(sm,rs):
    try: return _norm(sm.step(rs))
    except:
        try: return _norm(sm.step(_rpi(rs)))
        except: return CompatAction()


def _build(rs,action,fb,wm,fix,route,poi,seat,sc_name):
    lv=_level(action)
    ta2=_ta(rs); tb2=_tb(rs,wm); tc2=_tc(rs); rest=_rest(rs)
    sqi=fb.sqi
    hrv=float(fb.ecg.hrv_rmssd or 0)
    eda=float(fb.eda.scl_mean or 0) if hasattr(fb,'eda') and fb.eda.scl_mean else 0
    stroke=float(np.clip((max(0,30-hrv)/30)*0.5+min(1,eda/5)*0.3+(1-float(sqi.overall_confidence))*0.2,0,1))
    spd=float(np.random.normal(80,5)) if sc_name and sc_name!="normal" else float(np.random.normal(65,8))
    thr=round(max(0,min(1,np.random.normal(0.3,0.05))),3)
    brk=round(max(0,min(1,np.random.normal(0.05,0.02))),3)
    if tc2.get("detected"): brk=round(min(1,brk+0.6),3)
    return {
        "version":"4.3","ts":time.time(),"level":lv,
        "summary":_SUMMARY.get(lv,"Monitoring."),"reason":_reason(action),
        "poi_label":_POI_LABEL.get(lv,""),"escalate_911":lv=="ESCALATE",
        "task_a":ta2,"task_b":tb2,"task_c":tc2,
        "task_d":{"score":round(stroke,3),"label":"neuro_risk","reason":"HRV-EDA proxy -- not clinically validated"},
        "webcam":wm or {},"sqi":{"overall":round(float(sqi.overall_confidence),3),
            "abstain":bool(sqi.abstain),"ecg_quality":round(float(getattr(sqi,"ecg_quality",0) or 0),3),
            "eda_contact":round(float(getattr(sqi,"eda_contact",0) or 0),3),
            "resp_quality":round(float(getattr(sqi,"resp_quality",0) or 0),3),
            "motion_level":round(float(getattr(sqi,"motion_level",0) or 0),3),
            "seat_ecg_quality":round(float(getattr(sqi,"seat_ecg_quality",0) or 0),3)},
        "features":_feats(fb),
        "previews":{"ecg":_ds(getattr(fb.ecg,"samples",None),500),
                    "resp":_ds(getattr(fb,"respiration",None) if hasattr(fb,"respiration") else None,200)},
        "gps":_fix_d(fix),"route":_route_d(route),"poi":poi,"rest_guidance":rest,
        "policy":action.to_dict() if hasattr(action,"to_dict") else {},
        "seat_ecg":seat or {},
        "vehicle":{"speed_kph":round(spd,1),"throttle":thr,"brake":brk,"gear":"D",
                   "autopilot":lv=="ESCALATE","source":"sim"},
        "runtime":{"fusion":_RUNTIME.get("fusion"),"source":_RUNTIME.get("source"),
                   "scenario":sc_name or _scenario,"live_only":GD_LIVE_ONLY},
    }


def _seat_win(ws):
    if not GD_SEAT_ECG: return None,_SEAT_NODE.snapshot()
    snap=_SEAT_NODE.snapshot()
    if not snap.get("connected"): return None,snap
    ts=float(snap.get("last_packet_ts",0) or 0)
    if ts<=0 or (time.time()-ts)>GD_SEAT_STALE: return None,snap
    win=_SEAT_NODE.latest_window(ws)
    if len(win.ecg)<max(64,int(win.fs_hz*ws*0.6)): return None,snap
    return win,snap



import subprocess as _sp, threading as _thr

_LAST_VOICE_LEVEL = None
_VOICE_COOLDOWN = 30.0
_VOICE_ENABLED = os.getenv('GD_VOICE','1').strip().lower() in {'1','true','yes','on'}
_LAST_VOICE_LV: str = ''
_LAST_VOICE_TS = 0.0

def _speak(msg: str) -> None:
    """Non-blocking macOS voice alert."""
    if not _VOICE_ENABLED: return
    def _run():
        try:
            _sp.run(["say", "-v", "Samantha", "-r", "160", msg],
                    timeout=10, capture_output=True)
        except Exception:
            pass
    _thr.Thread(target=_run, daemon=True).start()

_VOICE_SCRIPTS = {
    "ADVISORY": "Attention. Early fatigue detected. Consider taking a short break soon.",
    "CAUTION":  "Warning. Fatigue confirmed. Please find a rest stop within the next few miles.",
    "PULLOVER": "Alert. Severe drowsiness detected. Pull over immediately and rest.",
    "ESCALATE": "Emergency. Medical event detected. Routing to nearest hospital. Emergency contacts notified.",
}

async def _pipeline():
    global _LAST_DISC
    fusion=FusionEngine(); sm=SafetyStateMachine()
    gps=RuntimeGPS(); nav_osm=OSMHospitalNav()
    nav_local=LocalHospitalNav(BASE_DIR/"data"/"hospitals.csv")
    _RUNTIME["fusion"]=fusion.status_dict()
    tel=ConsoleTelephony()
    dn=DiscordNotifier.from_env()
    sim=None; sim_iter=None; sc_name=None; last_lv=None

    while True:
        cur_sc=_scenario; wm=None
        if _CAM is not None:
            try:
                raw=_CAM.latest_metrics()
                wm=raw.to_dict() if hasattr(raw,"to_dict") else dict(raw)
            except: pass

        sw,seat=_seat_win(GD_WINDOW)
        fix=gps.read_fix(); route=_nav(fix,nav_osm,nav_local,'NOMINAL')

        if sw is not None:
            frame=SensorFrame(session_id="live",subject_id="live",
                              timestamp=time.monotonic(),window_sec=GD_WINDOW,
                              label=TaskLabel.UNKNOWN,webcam_metrics=wm)
            frame.seat_ecg=np.asarray(sw.ecg,dtype=np.float32)
            frame.seat_ecg_fs_hz=float(sw.fs_hz)
            frame.seat_ecg_meta=dict(sw.meta or {})
            _RUNTIME["source"]="seat_live"
            sim=None
        elif not GD_LIVE_ONLY:
            if sim is None or sc_name!=cur_sc:
                sim=GuardianSimulator(cur_sc,duration=99999.0,inject_artifacts=False)
                sim_iter=sim.stream(GD_WINDOW,GD_STEP)
                sc_name=cur_sc; _RUNTIME["source"]=f"sim:{cur_sc}"
            try:
                frame = next(sim_iter)
            except StopIteration:
                sim=None; sim_iter=None; await asyncio.sleep(0.1); continue
            except Exception:
                sim=None; sim_iter=None; await asyncio.sleep(0.5); continue
            if wm: frame.webcam_metrics=wm
        else:
            idle={"version":"4.3","ts":time.time(),"level":"INACTIVE",
                  "summary":"Waiting for live sensor. Set GD_LIVE_ONLY=0 for simulation.","reason":"No sensor",
                  "poi_label":"","escalate_911":False,
                  "task_a":{"score":0,"cls":"unknown","abstained":True,"hr_bpm":0,"confidence":0},
                  "task_b":{"score":0,"abstained":True},"task_c":{"score":0,"detected":False},
                  "task_d":{"score":0},"webcam":wm or {},"sqi":{"overall":0,"abstain":True},
                  "features":{},"previews":{"ecg":None,"resp":None},
                  "gps":_fix_d(fix),"route":_route_d(route),"poi":None,
                  "rest_guidance":None,"policy":{"level_name":"INACTIVE","escalate_911":False},
                  "seat_ecg":seat,"vehicle":{"speed_kph":0,"gear":"P","autopilot":False,"source":"none"},
                  "runtime":{"source":"idle","live_only":True}}
            async with _lock: _state.update(idle)
            await _broadcast(idle); await asyncio.sleep(max(0.25,GD_STEP)); continue

        sqi2=compute_sqi(frame); fb=extract_features(frame,sqi2,GD_WINDOW)
        if wm: fb.webcam_metrics=wm
        rs=fusion.run(fb); action=_rpol(sm,rs); lv=_level(action)
        poi_r=None; ams=_POI_MAP.get(lv)
        if ams and fix:
            try: poi_r=await asyncio.wait_for(_poi(fix.point.lat,fix.point.lon,ams),timeout=3.0)
            except: pass

        payload=_build(rs,action,fb,wm,fix,route,poi_r,seat,sc_name)
        async with _lock: _state.update(payload)

        # nuScenes real BEV frame
        _bev_frame = None
        if _NUSCENES is not None:
            try:
                _bev_frame = _NUSCENES.next_frame()
            except Exception:
                pass

        if _bev_frame:
            payload["bev"] = _bev_frame
        await _broadcast(payload)

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

        if lv=="ESCALATE":
            now=time.time()
            if (now-_LAST_DISC)>60.0:
                _LAST_DISC=now
                asyncio.create_task(_discord(payload))
                if dn and last_lv!="ESCALATE":
                    try: dn.send(title=f"GUARDIAN DRIVE -- ESCALATE",body=_reason(action),meta={"gps":_fix_d(fix)})
                    except: pass

        last_lv=lv
        await asyncio.sleep(max(0.1,GD_STEP))


@asynccontextmanager
async def lifespan(app:FastAPI):
    global _CAM
    bind_seat_node(_SEAT_NODE)
    print(f"\n{'='*60}\nGUARDIAN DRIVE(tm) v4.3\n  live_only={GD_LIVE_ONLY}  webcam={GD_WEBCAM}  seat={GD_SEAT_ECG}\n{'='*60}\n")
    if GD_WEBCAM and _WEBCAM_CLS:
        try: _CAM=_WEBCAM_CLS(cam_index=GD_WEBCAM_IDX); print(f"[GD] Webcam started cam {GD_WEBCAM_IDX}")
        except Exception as e: print(f"[GD] Webcam failed: {e}"); _CAM=None
    task=asyncio.create_task(_pipeline())
    try: yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError): await task
        if _CAM and hasattr(_CAM,"close"):
            try: _CAM.close()
            except: pass


app=FastAPI(title="Guardian Drive(tm)",version="4.3",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
app.mount("/static",StaticFiles(directory=str(STATIC_DIR)),name="static")
app.include_router(sensor_router)


@app.websocket("/ws")
async def ws_ep(ws:WebSocket):
    await ws.accept()
    _clients.add(ws)
    try:
        async with _lock: snap=dict(_state)
        if snap: await ws.send_text(json.dumps(snap,separators=(",",":")))
        while True: await ws.receive_text()
    except WebSocketDisconnect: pass
    except Exception: pass
    finally: _clients.discard(ws)


async def _sse_gen(req:Request)->AsyncGenerator[str,None]:
    while True:
        if await req.is_disconnected(): break
        async with _lock: snap=dict(_state)
        yield f"data: {json.dumps(snap,separators=(',',':'))}\n\n"
        await asyncio.sleep(0.5)

@app.get("/stream")
async def sse(req:Request):
    return StreamingResponse(_sse_gen(req),media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.get("/healthz")
def healthz(): return {"ok":True,"ts":time.time(),"version":"4.3"}

@app.get("/api/status")
async def status():
    async with _lock: return JSONResponse(_state)

@app.post("/api/scenario")
def set_sc(req:ScenarioReq):
    global _scenario
    if req.scenario in SCENARIO_PARAMS: _scenario=req.scenario; return {"ok":True,"scenario":req.scenario}
    return JSONResponse({"ok":False,"error":f"Unknown: {req.scenario}"},status_code=400)

@app.post("/api/gps")
def upd_gps(req:GpsReq):
    RuntimeGPS.set_fix(lat=req.lat,lon=req.lon,accuracy_m=req.accuracy_m,timestamp_unix=req.timestamp_unix)
    return {"ok":True}

@app.get("/api/poi")
async def get_poi(lat:float=37.7749,lon:float=-122.4194,level:str="CAUTION"):
    ams=_POI_MAP.get(level.upper(),["cafe"])
    if not ams: return JSONResponse({"poi":None})
    return JSONResponse({"poi":await _poi(lat,lon,ams)})

@app.post("/api/notify_discord")
async def notif_discord():
    if not GD_DISCORD: return JSONResponse({"ok":False,"error":"GD_DISCORD_WEBHOOK_URL not set"},status_code=400)
    async with _lock: snap=dict(_state)
    asyncio.create_task(_discord(snap)); return {"ok":True}

@app.get("/api/dispatch_script")
async def disp_script():
    async with _lock: s=dict(_state)
    gps=s.get("gps") or {}; route=s.get("route") or {}; poi=s.get("poi") or {}
    ta2=s.get("task_a") or {}; tb2=s.get("task_b") or {}; tc2=s.get("task_c") or {}
    return {"ok":True,"script":"\n".join([
        "GUARDIAN DRIVE(tm) EMERGENCY DISPATCH",
        f"Time: {time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}",
        f"Level: {s.get('level','--')}",f"Reason: {s.get('reason','--')}",
        f"GPS: {gps.get('lat','?')}, {gps.get('lon','?')}",f"Maps: {gps.get('maps','--')}",
        f"ER: {poi.get('name') or route.get('destination_name','--')} ({poi.get('distance_mi','?')} mi)",
        f"Task A ECG: cls={ta2.get('cls','--')} conf={ta2.get('confidence',0):.2f} HR={ta2.get('hr_bpm',0):.0f}bpm",
        f"Task B Drowsy: score={tb2.get('score',0):.2f}",
        f"Task C Crash: {'DETECTED' if tc2.get('detected') else 'none'} g={tc2.get('g_peak',0):.1f}",
        "","This system is not a medical device."])}

@app.get("/api/runtime_status")
def rt_status():
    return JSONResponse({"ok":True,"version":"4.3","live_only":GD_LIVE_ONLY,
                         "webcam":_CAM is not None,"seat_ecg":GD_SEAT_ECG,
                         "scenario":_scenario,"clients":len(_clients),
                         "fusion":_RUNTIME.get("fusion"),"source":_RUNTIME.get("source")})

@app.get("/camera.mjpg")
async def cam_mjpg():
    if _CAM is None: return JSONResponse({"ok":False,"error":"Webcam disabled"},status_code=400)
    async def gen():
        while True:
            jpg=_CAM.latest_jpeg()
            if jpg: yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "+str(len(jpg)).encode()+b"\r\n\r\n"+jpg+b"\r\n"
            await asyncio.sleep(0.08)
    return StreamingResponse(gen(),media_type="multipart/x-mixed-replace; boundary=frame")

def _srv(name):
    p=STATIC_DIR/name
    if p.exists(): return FileResponse(str(p))
    return HTMLResponse(f"<h3>{name} not found in server/static/</h3>",status_code=404)

@app.get("/")
def index(): return _srv("GuardianDrive_Dashboard.html")

@app.get("/gps_sender.html")
def gps_page(): return _srv("gps_sender.html")

@app.get("/preview.html")
def preview(): return _srv("preview.html")

@app.get("/favicon.ico")
def favicon(): return Response(status_code=204)

if __name__=="__main__":
    import uvicorn
    uvicorn.run("server.app:app",host=os.getenv("GD_HOST","0.0.0.0"),port=int(os.getenv("GD_PORT","8000")),reload=False)
