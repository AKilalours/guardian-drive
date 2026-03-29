# Guardian Drive v6.0 — Real-World Data & Integration Guide (Portfolio-Safe)

This repo is *integration-ready*. "Real" means:
- real-time pipeline with traceable decisions
- real data ingestion + replay
- pluggable providers for GPS / comms / vehicle control
- UI that shows live decisions and evidence
- honest claims (risk screening, not medical diagnosis)

## What you MUST NOT claim
- "Autopilot drives to hospital" on a real vehicle (OEM-only, regulated, liability)
- "Direct PSAP/911 integration" (requires certified eCall/NG911 pathway)
- "Medical-grade diagnosis" (FDA/CE)

## What you *can* claim (and recruiters respect)
- Designed a safety-critical edge inference pipeline with abstain behavior (SQI), persistent state machine, and integration interfaces
- Built a live dashboard + API streaming + replay logs for debugging and evaluation
- Implemented emergency contact notification (Discord/Twilio) + location/route advisory (demo-safe)
- Implemented **human-in-the-loop 911** (one-tap dial + generated dispatch script)

---

## Real Data: minimal v0.1 collection you can do safely

### A) ECG (serial streaming)
Hardware options:
- AD8232 + microcontroller (Arduino/ESP32) streaming samples over serial
- OpenBCI / BITalino streaming via serial/BT (depending on device)

Record:
```bash
python tools/list_serial_ports.py
python tools/record_serial_ecg.py --port /dev/tty.usbserial-XXXX --baud 115200 --seconds 600 --out data/raw/ecg_session.csv
python tools/build_windows_from_ecg_csv.py --csv data/raw/ecg_session.csv --fs 250 --window 30 --step 5 --out runs/ecg_replay.jsonl
```

Replay through the pipeline in the live dashboard (server):
```bash
export GD_REPLAY_JSONL=runs/ecg_replay.jsonl
python -m server.app
# open http://127.0.0.1:8000 and switch scenario to "artifact" if you want to stress SQI
```

### B) GPS (real)
- Easiest: use **browser geolocation** (dashboard toggle "GPS")
- Hardware: NMEA serial GPS module or gpsd:
  - `integrations/gps/gps_nmea_serial.py`
  - `integrations/gps/gps_gpsd.py`

### C) Emergency contact
- Default: console print
- Optional: Twilio SMS/call to your own phone (NOT 911):
  - set env vars in `telephony_twilio.py` docs

Free option (recommended): Discord webhook
```bash
export GD_DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
```

---

## Real nearby hospital routing (no dummy list)
The server now uses **OpenStreetMap Overpass API** to lookup nearby hospitals/ERs in real time.
If Overpass is unavailable (offline / rate limit), it falls back to `data/hospitals.csv`.

## If you want "autopilot to hospital" *for a portfolio*
Do it in a simulator, clearly:
- CARLA / LGSVL / AirSim destination routing demo
- Your code already has the `VehicleControlProvider` interface; implement `CarlaVehicleControl`
- Claim: **"closed-loop routing in simulation"**, not real vehicle autopilot

## Human-in-the-loop 911
This repo intentionally does **not** auto-dial 911.
Instead, the dashboard provides:
- `CALL 911` one-tap dial (`tel:911`)
- `Copy Dispatch Script` button (auto-generated text + GPS + nearest ER)

This is the safest and most credible way to demonstrate emergency response.
