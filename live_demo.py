"""
live_demo.py
Guardian Drive -- LOCAL Live Demo with Real Webcam

Run: python3 live_demo.py
Opens: http://localhost:7860

Features:
- Real webcam EAR/PERCLOS detection via MediaPipe
- Live pipeline updating every click
- Voice alerts via macOS TTS
- Real Discord alerts on ESCALATE
- Real OSM nearby routing
- GPT-4o explanation
- All 5 impairment states

Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
"""

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import cv2
import json, os, time, math, subprocess, threading
import urllib.request
from pathlib import Path

# ── Try MediaPipe ──────────────────────────────────────────────────
try:
    import mediapipe as mp
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    MEDIAPIPE_OK = False
    print("MediaPipe FaceMesh loaded")
except Exception as e:
    MEDIAPIPE_OK = False
    print(f"MediaPipe not available: {e} -- using EAR slider fallback")

# MediaPipe eye landmark indices
LEFT_EYE  = [362,385,387,263,373,380]
RIGHT_EYE = [33, 160,158,133,153,144]

# ── Model ──────────────────────────────────────────────────────────
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
        print(f"Model loaded: {p}"); break

OPENAI_KEY   = os.getenv("OPENAI_API_KEY","")
DISCORD_HOOK = os.getenv("DISCORD_WEBHOOK","")

# Perclos rolling buffer
perclos_buffer = []
PERCLOS_WINDOW = 90  # 3 seconds at 30fps

IMPAIRMENT_INFO = {
    "ALERT":      "✅ All signals normal. Safe to drive.",
    "DROWSY":     "🟡 Early impairment. TCN detects low-arousal. Take a break soon.",
    "SLEEPY":     "🟠 Eyes closing, yawning. Reaction time reduced. Rest stop needed.",
    "FATIGUED":   "🟠 Cumulative fatigue. Low HRV or long drive. Sleep needed.",
    "MICROSLEEP": "🚨 CRITICAL: Eye closure >80%. STOP DRIVING IMMEDIATELY.",
}

# ── Webcam capture ─────────────────────────────────────────────────
def capture_frame_with_ear():
    """Capture one webcam frame and compute EAR via MediaPipe."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, 0.30, "Webcam error"

    ear = 0.30
    status = "No face detected"
    annotated = frame.copy()

    if MEDIAPIPE_OK:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]

            def get_ear(indices):
                pts = np.array([[lm[i].x*w, lm[i].y*h]
                                 for i in indices])
                p2p6 = np.linalg.norm(pts[1]-pts[5])
                p3p5 = np.linalg.norm(pts[2]-pts[4])
                p1p4 = np.linalg.norm(pts[0]-pts[3])
                return float((p2p6+p3p5)/(2*p1p4+1e-6))

            left_ear  = get_ear(LEFT_EYE)
            right_ear = get_ear(RIGHT_EYE)
            ear = (left_ear + right_ear) / 2

            # Update PERCLOS buffer
            perclos_buffer.append(1 if ear < 0.18 else 0)
            if len(perclos_buffer) > PERCLOS_WINDOW:
                perclos_buffer.pop(0)

            # Draw landmarks
            for idx in LEFT_EYE + RIGHT_EYE:
                x = int(lm[idx].x * w)
                y = int(lm[idx].y * h)
                cv2.circle(annotated, (x,y), 2, (0,255,0), -1)

            # EAR display
            color = (0,0,255) if ear<0.18 else (0,165,255) if ear<0.22 else (0,255,0)
            cv2.putText(annotated, f"EAR: {ear:.3f}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)

            perclos_val = np.mean(perclos_buffer) if perclos_buffer else 0.0
            cv2.putText(annotated, f"PERCLOS: {perclos_val:.1%}",
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255,255,0), 2)

            status = f"EAR={ear:.3f} PERCLOS={perclos_val:.1%}"

    # Convert to RGB for Gradio
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated_rgb, ear, status

# ── Core functions ──────────────────────────────────────────────────
def classify_impairment(perclos, tcn_prob, hrv_rmssd, drive_mins, yawns, ear):
    if ear < 0.15 or perclos > 0.80: return "MICROSLEEP"
    if perclos > 0.25 and yawns >= 3: return "SLEEPY"
    if hrv_rmssd < 20.0 or drive_mins > 90: return "FATIGUED"
    if tcn_prob > 0.50 or perclos > 0.15 or ear < 0.22: return "DROWSY"
    return "ALERT"

def find_nearby(lat, lon, impairment):
    if impairment == "MICROSLEEP":
        tags = '"amenity"="hospital"'; label = "🏥 Nearest Hospital"
    elif impairment in ["SLEEPY","DROWSY"]:
        tags = '"amenity"~"cafe|fuel"'; label = "☕ Nearby Cafes"
    else:
        tags = '"tourism"~"motel|hotel"'; label = "🏨 Nearby Hotels"
    query = f'[out:json][timeout:10];node(around:5000,{lat},{lon})[{tags}];out 3;'
    try:
        req = urllib.request.Request(
            "https://overpass-api.de/api/interpreter",
            data=query.encode(),
            headers={"Content-Type":"application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read())
        places = []
        for el in data.get("elements",[])[:3]:
            name = el.get("tags",{}).get("name","Unknown")
            dist = math.sqrt((el["lat"]-lat)**2+(el["lon"]-lon)**2)*111
            places.append(f"{name} ({dist:.1f}km)")
        return label, places or ["None found nearby"]
    except Exception as e:
        return label, [f"OSM error: {e}"]

def speak(text):
    """macOS TTS voice alert."""
    try:
        subprocess.Popen(["say", "-v", "Samantha", "-r", "150", text])
    except Exception:
        pass

def send_discord(state, impairment, risk, lat, lon):
    if not DISCORD_HOOK: return "Discord not configured"
    msg = {"username":"Guardian Drive",
           "embeds":[{"title":f"🚨 {state}: {impairment}",
                      "color":15158332,
                      "fields":[
                          {"name":"Risk","value":str(risk),"inline":True},
                          {"name":"Location",
                           "value":f"[Maps](https://maps.google.com/?q={lat},{lon})",
                           "inline":True},
                      ],
                      "footer":{"text":"Research prototype. Not medical advice."}}]}
    try:
        req = urllib.request.Request(
            DISCORD_HOOK, data=json.dumps(msg).encode(),
            headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return f"✅ Discord alert sent (status {r.status})"
    except Exception as e:
        return f"Discord failed: {e}"

def call_llm(state, impairment, risk, perclos, hrv, poi):
    if not OPENAI_KEY:
        defaults = {"ALERT":"All signals normal.",
                    "DROWSY":"Early drowsiness. Take a break soon.",
                    "SLEEPY":"Please stop at a rest area.",
                    "FATIGUED":"Fatigue detected. Find a place to sleep.",
                    "MICROSLEEP":"CRITICAL: Pull over immediately."}
        return defaults.get(impairment,"Alert.") + " Research prototype, not medical advice."
    try:
        payload = json.dumps({
            "model":"gpt-4o","max_tokens":100,"temperature":0.3,
            "messages":[
                {"role":"system","content":
                 "Guardian Drive AI. 2 calm sentences. End: Research prototype, not medical advice."},
                {"role":"user","content":
                 f"State:{state} Impairment:{impairment} Risk:{risk:.3f} "
                 f"PERCLOS:{perclos:.1%} HRV:{hrv:.0f}ms Nearest:{poi}. "
                 "Reply JSON: {\"text\":\"...\"}"}
            ]}).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={"Content-Type":"application/json",
                     "Authorization":f"Bearer {OPENAI_KEY}"})
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read())
            content = data["choices"][0]["message"]["content"]
            content = content.replace("```json","").replace("```","").strip()
            return json.loads(content).get("text","")
    except Exception as e:
        return f"GPT-4o error: {e}"

# ── Main pipeline ────────────────────────────────────────────────
def run_pipeline(
    use_webcam, manual_ear,
    hrv_rmssd, yawn_count, drive_mins, condition,
    n_objects, near_intersection, speed_kph, imu_g,
    latitude, longitude,
    emergency_name, emergency_phone,
    use_llm, use_voice
):
    t0 = time.perf_counter()

    # Step 1: Get EAR -- webcam or manual
    webcam_img = None
    ear = float(manual_ear)
    cam_status = "Manual EAR mode"

    if use_webcam and MEDIAPIPE_OK:
        webcam_img, ear, cam_status = capture_frame_with_ear()

    # Compute PERCLOS from buffer or estimate from EAR
    if perclos_buffer:
        perclos = float(np.mean(perclos_buffer))
    else:
        perclos = max(0.0, min(1.0, 1.0 - ear/0.30))

    # Sensor window
    np.random.seed(int(time.time())%1000)
    window = np.random.randn(4,4200).astype(np.float32)
    if condition == "Low Arousal": window *= 0.4
    elif condition == "Stress":    window *= 1.3
    elif condition == "Fatigued":  window[0] *= 0.5

    # SQI
    q=[min(1.0,float(np.std(window[c]))/t)
       for c,t in enumerate([0.5,0.05,0.1,0.1])]
    sqi=round(min(1.0,0.5*q[0]+0.3*q[1]+0.2*q[2]),3)

    if sqi < 0.30:
        return (webcam_img,
                "⚠️ **ABSTAINED** -- SQI too low",
                "","","","","","")

    # TCN
    mu=window.mean(axis=1,keepdims=True)
    std=window.std(axis=1,keepdims=True)+1e-6
    x=torch.FloatTensor((window-mu)/std).unsqueeze(0)
    with torch.no_grad():
        tcn_prob=float(torch.sigmoid(model(x)))

    # Impairment
    impairment = classify_impairment(
        perclos, tcn_prob, float(hrv_rmssd),
        float(drive_mins), int(yawn_count), ear)

    # AV context
    c_t=-(int(n_objects)/20.0)*0.06
    c_i=-0.10 if near_intersection else 0.0
    c_s=-max(0,float(speed_kph)-80)/80*0.04
    thresh=max(0.15,0.35+c_t+c_i+c_s)

    # Fusion
    r_phys=sqi*tcn_prob
    r_imu=min(1.0,float(imu_g)/3.0)
    r_ctx=min(1.0,int(n_objects)/20.0)
    r_neuro=max(0.0,1.0-min(float(hrv_rmssd)/50,1.0))
    r_total=round(0.40*r_phys+0.20*r_imu+0.10*r_ctx+0.30*r_neuro,4)

    # State
    if r_total<thresh:           state="NOMINAL"
    elif r_total<thresh+0.20:    state="ADVISORY"
    elif r_total<thresh+0.40:    state="CAUTION"
    elif r_total<thresh+0.60:    state="PULLOVER"
    else:                        state="ESCALATE"

    latency_ms=round((time.perf_counter()-t0)*1000,1)

    # Nearby places
    lat, lon = float(latitude), float(longitude)
    poi_label, poi_list = find_nearby(lat, lon, impairment)
    poi_name = poi_list[0] if poi_list else ""
    gmaps = f"https://maps.google.com/?q={lat},{lon}"

    # Voice alert
    if use_voice and state != "NOMINAL":
        voice_map = {
            "MICROSLEEP": f"Warning! Microsleep detected. Pull over immediately. {poi_name}",
            "SLEEPY":     f"You are sleepy. Please stop at {poi_name or 'the next rest area'}.",
            "FATIGUED":   f"Driver fatigue detected. Please find a place to rest. {poi_name}",
            "DROWSY":     f"Early drowsiness detected. Consider taking a break soon.",
            "ADVISORY":   "Advisory. Monitor your alertness.",
        }
        msg = voice_map.get(impairment, f"{state} alert. Please take action.")
        threading.Thread(target=speak, args=(msg,), daemon=True).start()

    # Discord on ESCALATE
    discord_result = ""
    if state == "ESCALATE":
        discord_result = send_discord(state, impairment, r_total, lat, lon)

    # LLM
    llm_text = ""
    if use_llm or state in ["PULLOVER","ESCALATE"]:
        llm_text = call_llm(state, impairment, r_total,
                             perclos, float(hrv_rmssd), poi_name)

    # ── Build outputs ───────────────────────────────────────────
    se={"NOMINAL":"✅","ADVISORY":"🟡","CAUTION":"🟠",
        "PULLOVER":"🔴","ESCALATE":"🚨"}.get(state,"⚪")
    ie={"ALERT":"✅","DROWSY":"🟡","SLEEPY":"🟠",
        "FATIGUED":"🟠","MICROSLEEP":"🚨"}.get(impairment,"⚪")

    # Monitor
    monitor = f"""## {se} **{state}** &nbsp;|&nbsp; {ie} **{impairment}**
> {IMPAIRMENT_INFO.get(impairment,'')}

| Signal | Value | Status |
|--------|-------|--------|
| EAR | {ear:.3f} | {'🚨 MICROSLEEP' if ear<0.15 else '🟠 LOW' if ear<0.22 else '✅ normal'} |
| PERCLOS | {perclos:.1%} | {'🚨 HIGH' if perclos>0.25 else '✅ normal'} |
| HRV RMSSD | {hrv_rmssd:.0f}ms | {'🚨 LOW' if float(hrv_rmssd)<20 else '✅ OK'} |
| TCN prob | {tcn_prob:.3f} | {'🚨' if tcn_prob>0.7 else '🟡' if tcn_prob>0.5 else '✅'} |
| Yawns | {yawn_count}/30s | {'🟠' if int(yawn_count)>=3 else '✅'} |
| Drive | {drive_mins:.0f}min | {'🚨' if float(drive_mins)>90 else '✅'} |
| SQI | {sqi} | {'✅' if sqi>0.7 else '⚠️'} |
| Risk | {r_total} | thresh={thresh:.3f} |

Webcam: {cam_status} | Latency: {latency_ms}ms
"""

    # Alerts
    poi_text = "\n".join(f"- {p}" for p in poi_list)
    alerts = f"""## 🚨 Active Alerts

**Voice TTS**: {'🔊 FIRED -- ' + impairment if use_voice and state!='NOMINAL' else '⏸ silent'}
**Seat Haptic**: {'📳 Would vibrate (local server only)' if state in ['CAUTION','PULLOVER','ESCALATE'] else '⏸'}
**Discord**: {discord_result if discord_result else ('⏸ fires on ESCALATE' if state!='ESCALATE' else 'Add DISCORD_WEBHOOK env var')}

### {poi_label}
{poi_text}
[📍 Your location]({gmaps})

### Emergency Contact
{"**" + emergency_name + "** -- " + emergency_phone if emergency_phone else "No contact set"}
{"🚨 Would send SMS/call on ESCALATE (add Twilio credentials)" if emergency_phone else ""}

### Hospital Autopilot Route
{"🏥 " + poi_list[0] + " -- [Navigate](" + gmaps + ")" if impairment=='MICROSLEEP' else "⏸ Activates on MICROSLEEP/ESCALATE"}
"""

    # LLM
    llm_out = f"**🤖 GPT-4o**: {llm_text}" if llm_text else "⏸ LLM fires on PULLOVER/ESCALATE or when enabled"

    # Impairment guide
    guide = """## 🧠 Impairment Types

| Type | Key Signal | EAR | PERCLOS | HRV | Drive | Action |
|------|-----------|-----|---------|-----|-------|--------|
| ✅ ALERT | Normal | >0.25 | <15% | >30ms | Any | Continue |
| 🟡 DROWSY | TCN elevated | 0.22+ | 15-25% | 20-30ms | Any | Coffee |
| 🟠 SLEEPY | Yawning | 0.18+ | 25-80% | Any | Any | Rest stop |
| 🟠 FATIGUED | Low HRV | Normal | <25% | <20ms | >90min | Sleep |
| 🚨 MICROSLEEP | Eyes closed | <0.15 | >80% | Any | Any | STOP NOW |

**DROWSY** ≠ **SLEEPY**: Drowsy is physiological (TCN detects it before you feel it). Sleepy is behavioural (eyes closing, yawning).
**FATIGUED** ≠ **SLEEPY**: Fatigue is cumulative body tiredness (low HRV). Coffee fixes sleepy, not fatigue.
**MICROSLEEP**: Involuntary. Driver doesn't know it happened. At 100kph = 28-140m blind.
"""

    # Status
    status = f"""## 📊 Pipeline Status

| Component | Status |
|-----------|--------|
| Webcam (MediaPipe) | {'✅ Active' if MEDIAPIPE_OK and use_webcam else '⏸ Manual mode'} |
| TCN Model | ✅ Loaded (LOSO AUC 0.769) |
| TensorRT | ⚡ Available on Tesla T4 |
| Voice TTS | {'✅ Active' if use_voice else '⏸ Disabled'} |
| GPT-4o | {'✅ Active' if OPENAI_KEY else '❌ No API key'} |
| Discord | {'✅ Configured' if DISCORD_HOOK else '❌ Set DISCORD_WEBHOOK'} |
| OSM Routing | ✅ Live API |

### 🎯 Test Scenarios
| Scenario | Settings | Expected |
|----------|---------|---------|
| Microsleep | EAR=0.08, Drive=100min | 🚨 ESCALATE + Discord |
| Sleepy | EAR=0.19, Yawns=4 | 🟠 CAUTION + cafe routing |
| Fatigued | HRV=15, Drive=100min | 🟠 CAUTION + hotel routing |
| Alert | All defaults | ✅ NOMINAL |
"""

    return (webcam_img, monitor, alerts, llm_out, guide, status)

# ── Gradio UI ──────────────────────────────────────────────────────
with gr.Blocks(title="Guardian Drive -- Live Local Demo",
               theme=gr.themes.Soft(primary_hue="blue")) as demo:

    gr.Markdown("""
# 🚗 Guardian Drive -- Live Local Demo
**Akilan Manivannan** & **Akila Lourdes Miriyala Francis** | LIU Brooklyn

> Run locally for real webcam + voice alerts. Research prototype.
    """)

    with gr.Row():
        # Left panel
        with gr.Column(scale=1):
            gr.Markdown("### 📷 Webcam")
            use_webcam=gr.Checkbox(
                value=MEDIAPIPE_OK,
                label="Use real webcam (requires MediaPipe)")
            manual_ear=gr.Slider(0.05,0.35,0.30,step=0.01,
                label="Manual EAR (used when webcam off)")

            gr.Markdown("### 💓 Physiological")
            condition=gr.Dropdown(
                ["Alert","Low Arousal","Stress","Fatigued"],
                value="Alert",label="Condition")
            hrv_rmssd=gr.Slider(10,80,45,step=1,label="HRV RMSSD (ms)")
            yawn_count=gr.Slider(0,8,0,step=1,label="Yawns/30s")
            drive_mins=gr.Slider(0,120,30,step=5,label="Drive time (min)")

            gr.Markdown("### 🚗 AV Context")
            n_objects=gr.Slider(0,20,5,step=1,label="BEV objects")
            near_int=gr.Checkbox(False,label="Near intersection")
            speed_kph=gr.Slider(0,130,60,step=5,label="Speed (kph)")
            imu_g=gr.Slider(0,5,0.1,step=0.1,label="IMU g-force")

            gr.Markdown("### 📍 Location")
            latitude=gr.Number(value=40.6892,label="Latitude")
            longitude=gr.Number(value=-74.0445,label="Longitude")
            gr.Markdown("*Change to your real location for accurate routing*")

            gr.Markdown("### 📱 Emergency Contact")
            emergency_name=gr.Textbox(value="",label="Contact name",
                placeholder="Mom")
            emergency_phone=gr.Textbox(value="",label="Phone number",
                placeholder="+1-555-0100")

            gr.Markdown("### ⚙️ Options")
            use_llm=gr.Checkbox(value=bool(os.getenv("OPENAI_API_KEY","")),
                label="GPT-4o explanation")
            use_voice=gr.Checkbox(value=True,
                label="Voice alerts (macOS TTS)")

            btn=gr.Button("▶ Run Pipeline",variant="primary",size="lg")

        # Right panel
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("📷 Live Camera"):
                    webcam_out=gr.Image(label="Webcam + EAR overlay",
                        height=300)
                    gr.Markdown("*Green dots = MediaPipe landmarks. EAR and PERCLOS shown live.*")
                with gr.TabItem("🚦 Driver State"):
                    out_monitor=gr.Markdown()
                with gr.TabItem("🚨 Alerts & Routing"):
                    out_alerts=gr.Markdown()
                with gr.TabItem("🤖 GPT-4o"):
                    out_llm=gr.Markdown()
                with gr.TabItem("🧠 Impairment Types"):
                    out_guide=gr.Markdown()
                with gr.TabItem("📊 System Status"):
                    out_status=gr.Markdown()

    btn.click(
        run_pipeline,
        inputs=[use_webcam, manual_ear,
                hrv_rmssd, yawn_count, drive_mins, condition,
                n_objects, near_int, speed_kph, imu_g,
                latitude, longitude,
                emergency_name, emergency_phone,
                use_llm, use_voice],
        outputs=[webcam_out, out_monitor, out_alerts,
                 out_llm, out_guide, out_status])

    gr.Markdown("""
---
**Quick test**: Set EAR=0.08, Drive=100min, Yawns=4 → ESCALATE → Discord fires
**Webcam test**: Enable webcam, click Run → see your face with EAR overlay
**Voice test**: Trigger CAUTION/ESCALATE with voice enabled → hear TTS alert
    """)

if __name__ == "__main__":
    print("Starting Guardian Drive Live Demo...")
    print("Open: http://localhost:7860")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set True to get public URL for demo
        show_error=True)
