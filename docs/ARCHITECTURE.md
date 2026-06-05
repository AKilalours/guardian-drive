# Guardian Drive -- Technical Architecture

## Pipeline

    Sensor Frame (8s window)
        -> SQI [abstain if noisy]
        -> Feature Bundle
        -> Task A + Task B + Task C + Task D (parallel)
        -> FusionEngine.run() -> RiskState
        -> SafetyStateMachine.step() -> PolicyAction
        -> POI lookup (OSM Overpass API)
        -> nuScenes BEV frame injection
        -> WebSocket broadcast -> Dashboard
        -> Voice alert + Discord webhook (if ESCALATE)

## State Machine

    INACTIVE -> NOMINAL -> ADVISORY -> CAUTION -> PULLOVER -> ESCALATE

    ADVISORY:  Task B > 0.35 or PERCLOS > 0.15 -- nearest cafe
    CAUTION:   Task B > 0.55 or PERCLOS > 0.25 -- nearest motel
    PULLOVER:  Task B > 0.75 or sustained CAUTION -- pull over now
    ESCALATE:  Cardiac event or crash -- hospital routing + Discord + 911

## Key Components

    sqi/compute.py              Signal quality gating
    features/extract.py         ECG/EDA/IMU/camera feature extraction
    policy/fusion.py            Multi-task fusion engine
    policy/state_machine.py     Safety state machine
    server/app.py               FastAPI WebSocket server
    acquisition/nuscenes_bev.py Real nuScenes BEV loader
    learned/waypoint_transformer.py GPT-2 trajectory predictor
    integrations/vision_webcam.py   MediaPipe FaceMesh EAR/PERCLOS/yawn

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| SQI gating | Prevents false positives from noisy signals |
| TCN over LSTM | Better parallelism, no vanishing gradient |
| Causal transformer | Same architecture as Tesla FSD neural planner |
| OSM over Google Maps | Free, no API key, works offline |
| WebSocket over polling | 50ms vs 500ms latency |
| Simulator fallback | Full demo without hardware |
