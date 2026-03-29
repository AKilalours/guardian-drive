# Repo Adaptation Plan

## Goal
Integrate seat-based ECG ingestion into the existing Guardian Drive repo without pretending the system is production-ready.

## Files added
- acquisition/seat_ecg_node.py
- acquisition/ring_buffer.py
- sqi/contact_quality.py
- sqi/motion_artifact.py
- sqi/window_quality.py
- features/ecg_filter.py
- features/rpeak_detect.py
- features/hrv_live.py
- features/resp_estimate.py
- server/routes_sensor.py
- config/seat_runtime.yaml

## Next required patch points
1. `features/extract.py`
   - consume `SeatECGNode.latest_window(...)`
   - compute quality, filtered signal, peaks, HRV
   - populate your existing `FeatureBundle.ecg`

2. `main.py`
   - instantiate `SeatECGNode`
   - bind it to the server route with `bind_seat_node(node)`
   - replace or augment simulator path with live seat ECG windowing

3. `policy/fusion.py`
   - make sure Task A / Task B can consume live HR / HRV / SQI from the seat path

4. `ui/`
   - add seat ECG connection status panel
   - add signal quality indicator
   - add live waveform preview only when SQI is usable

## Honest claims after this stage
- live seat ECG ingestion path
- real-time multimodal prototype
- signal-quality-aware runtime pipeline

## Claims still not honest
- medical grade
- cardiac arrest detection
- production-ready emergency system
- real vehicle control
