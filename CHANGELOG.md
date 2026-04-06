# Guardian Drive -- Changelog

## v4.3 (2026-04-06)
- GPT-2 style waypoint transformer trained on nuScenes (ADE 7.70m)
- Real nuScenes BEV loader (18,538 annotations, 10 scenes)
- Voice alerts via macOS say command
- Real GPS + OpenStreetMap POI routing
- Inline Google Maps panel with route display
- Discord auto-fire on ESCALATE
- EAR/PERCLOS/yawn calibrated to driver face geometry

## v4.2
- WESAD TCN training complete -- AUC 0.9514 (epoch 11/15)
- WebSocket dashboard with real-time BEV
- nuScenes ego pose matching by timestamp

## v4.1
- PTB-XL Task A arrhythmia screener
- Safety state machine NOMINAL to ESCALATE
- MediaPipe FaceMesh integration

## v4.0
- Initial multimodal pipeline
- SQI gating before inference
- FastAPI WebSocket server
