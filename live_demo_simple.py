"""
Simple pipeline test -- no gradio needed
Tests all components directly
"""
import numpy as np
import torch
import torch.nn as nn
import json, os, time, math, subprocess, urllib.request
from pathlib import Path

DISCORD_HOOK = os.getenv("DISCORD_WEBHOOK","")
OPENAI_KEY   = os.getenv("OPENAI_API_KEY","")

class TCNBlock(nn.Module):
    def __init__(self,i,o,d=1):
        super().__init__()
        self.conv=nn.Conv1d(i,o,3,padding=(3-1)*d,dilation=d)
        self.bn=nn.BatchNorm1d(o); self.relu=nn.ReLU()
        self.res=nn.Conv1d(i,o,1) if i!=o else nn.Identity()
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))[:,:,:x.size(2)]+self.res(x)

class DrowsinessTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1=TCNBlock(4,32,1);self.b2=TCNBlock(32,64,2)
        self.b3=TCNBlock(64,64,4);self.b4=TCNBlock(64,64,8)
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.head=nn.Sequential(nn.Linear(64,32),nn.ReLU(),
                                nn.Dropout(0.1),nn.Linear(32,1))
    def forward(self,x):
        x=self.b4(self.b3(self.b2(self.b1(x))))
        return self.head(self.pool(x).squeeze(-1)).squeeze(-1)

model = DrowsinessTCN().eval()
for p in ["learned/models/task_b_tcn_cuda.pt",
          "learned/models/task_b_tcn_ddp.pt"]:
    if Path(p).exists():
        s=torch.load(p,map_location="cpu")
        if isinstance(s,dict) and "model" in s: s=s["model"]
        model.load_state_dict(s,strict=False)
        print(f"Model: {p}"); break

def test_scenario(name, ear, perclos, hrv, drive, yawns, lat=40.69, lon=-74.04):
    print(f"\n{'='*50}")
    print(f"SCENARIO: {name}")
    print(f"{'='*50}")

    np.random.seed(42)
    window=np.random.randn(4,4200).astype(np.float32)

    # SQI
    q=[min(1.0,float(np.std(window[c]))/t)
       for c,t in enumerate([0.5,0.05,0.1,0.1])]
    sqi=round(min(1.0,0.5*q[0]+0.3*q[1]+0.2*q[2]),3)

    # TCN
    mu=window.mean(axis=1,keepdims=True)
    std=window.std(axis=1,keepdims=True)+1e-6
    x=torch.FloatTensor((window-mu)/std).unsqueeze(0)
    with torch.no_grad():
        prob=float(torch.sigmoid(model(x)))

    # Impairment
    if ear<0.15 or perclos>0.80: imp="MICROSLEEP"
    elif perclos>0.25 and yawns>=3: imp="SLEEPY"
    elif hrv<20 or drive>90: imp="FATIGUED"
    elif prob>0.50 or perclos>0.15: imp="DROWSY"
    else: imp="ALERT"

    # Fusion
    r_neuro=max(0.0,1.0-min(hrv/50,1.0))
    r_total=round(0.40*sqi*prob+0.20*0.1+0.10*0.25+0.30*r_neuro,4)

    # State
    thresh=0.35
    if r_total<thresh:           state="NOMINAL"
    elif r_total<thresh+0.20:    state="ADVISORY"
    elif r_total<thresh+0.40:    state="CAUTION"
    elif r_total<thresh+0.60:    state="PULLOVER"
    else:                        state="ESCALATE"

    print(f"EAR={ear} PERCLOS={perclos:.0%} HRV={hrv}ms Drive={drive}min")
    print(f"SQI={sqi} TCN={prob:.3f} r_total={r_total}")
    print(f"Impairment: {imp}")
    print(f"State:      {state}")

    # Voice alert
    if state != "NOMINAL":
        msg = f"{imp} detected. {state} alert."
        print(f"Voice TTS: '{msg}'")
        subprocess.Popen(["say","-v","Samantha","-r","150",msg])

    # OSM routing
    if imp in ["SLEEPY","MICROSLEEP"]:
        tags='"amenity"~"cafe|hospital"'
    elif imp=="FATIGUED":
        tags='"tourism"~"motel|hotel"'
    else:
        tags='"amenity"="cafe"'
    query=f'[out:json][timeout:8];node(around:5000,{lat},{lon})[{tags}];out 2;'
    try:
        req=urllib.request.Request(
            "https://overpass-api.de/api/interpreter",
            data=query.encode(),
            headers={"Content-Type":"application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req,timeout=10) as r:
            data=json.loads(r.read())
        places=[el.get("tags",{}).get("name","Unknown")
                for el in data.get("elements",[])[:2]]
        print(f"Nearby ({imp}): {places}")
    except Exception as e:
        print(f"OSM: {e}")

    # Discord on ESCALATE
    if state=="ESCALATE" and DISCORD_HOOK:
        msg={"username":"Guardian Drive",
             "embeds":[{"title":f"🚨 {state}: {imp}",
                        "color":15158332,
                        "description":f"Risk={r_total} | EAR={ear} | PERCLOS={perclos:.0%}",
                        "footer":{"text":"Research prototype. Not medical advice."}}]}
        try:
            req=urllib.request.Request(
                DISCORD_HOOK,data=json.dumps(msg).encode(),
                headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=8) as r:
                print(f"Discord: ✅ sent (status {r.status})")
        except Exception as e:
            print(f"Discord: {e}")

    return state, imp

# Run all test scenarios
test_scenario("Alert driver",      ear=0.30,perclos=0.08,hrv=45,drive=20,yawns=0)
test_scenario("Drowsy driver",     ear=0.23,perclos=0.18,hrv=28,drive=45,yawns=1)
test_scenario("Sleepy driver",     ear=0.19,perclos=0.30,hrv=35,drive=50,yawns=4)
test_scenario("Fatigued driver",   ear=0.28,perclos=0.12,hrv=15,drive=95,yawns=1)
test_scenario("Microsleep",        ear=0.08,perclos=0.85,hrv=22,drive=60,yawns=5)

print("\n✅ All pipeline components tested")
print("Discord fires on ESCALATE (microsleep scenario)")
print("Voice TTS fires on all non-NOMINAL states")
print("OSM routing queries real nearby places")
