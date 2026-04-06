# Contributing to Guardian Drive

## Team
- Akila Lourdes Miriyala Francis 
- Akilan Manivannan 

## Setup
    git clone https://github.com/AKilalours/guardian-drive.git
    cd guardian-drive
    cp .env.example .env
    pip install -r requirements.txt

## Run
    GD_LIVE_ONLY=0 GD_ENABLE_WEBCAM=1 GD_WINDOW_SEC=8 GD_STEP_SEC=2
    GD_TASK_B_MLP_WEIGHTS=learned/models/task_b_tcn.pt GD_VOICE=1
    python -m uvicorn server.app:app --host 127.0.0.1 --port 8000

## Test Pipeline
    python acquisition/nuscenes_bev.py
    python learned/waypoint_transformer.py --infer

## Safety Notice
This is a research project. Not for use in production vehicles.
All changes must preserve the medical_grade: false claim guardrail.
