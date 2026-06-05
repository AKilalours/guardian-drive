# Guardian Drive — MLOps & Model Lifecycle

> Built by **Akila Lourdes Miriyala Francis** & **Akilan Manivannan**

---

## Model Registry

| Model | Path | Framework | Result | Training Data | Owner |
|-------|------|-----------|--------|---------------|-------|
| Task B TCN | learned/models/task_b_tcn.pt | PyTorch | AUC 0.9514 | WESAD 15 subjects | Akila |
| Task A RF | artifacts/task_a_model_real_ptbxl.joblib | Scikit-learn | Real PTB-XL | Clinical ECG | Akila |
| Waypoint Transformer | learned/models/waypoint_transformer.pt | PyTorch | ADE 7.70m | nuScenes 31206 poses | Akilan |
| Drowsiness CNN | artifacts/drowsiness_cnn_subjectsplit.pt | PyTorch | Baseline | Subject-split | Akilan |

---

## Training Pipelines

### Task B — WESAD TCN (Akila Lourdes Miriyala Francis)

    python learned/task_b_trainer.py --data_dir datasets/WESAD/WESAD

    Dataset:  WESAD 15 subjects, 2874 windows
    Split:    80/20 random
    Epochs:   15
    Best:     AUC 0.9514 at epoch 11
    Output:   learned/models/task_b_tcn.pt
    Results:  learned/results/task_b_eval.json

Training curve:

| Epoch | Loss   | AUC    |
|-------|--------|--------|
| 1     | 0.3867 | 0.8330 |
| 5     | 0.2917 | 0.9212 |
| 8     | 0.2749 | 0.9382 |
| 11    | 0.2295 | 0.9514 | <- best
| 15    | 0.2227 | 0.9482 |

### Task A — PTB-XL (Akila Lourdes Miriyala Francis)

    pip install wfdb
    python learned/task_a_trainer.py --data_dir datasets/ptbdb/1.0.0

    Dataset:  PTB-XL PhysioNet real clinical ECG
    Model:    RandomForest + 1D CNN FFT features
    Output:   artifacts/task_a_model_real_ptbxl.joblib

### Waypoint Transformer — GPT-2 (Akilan Manivannan)

    python learned/waypoint_transformer.py --train

    Dataset:  nuScenes mini 31206 real ego poses
    Model:    Causal self-attention transformer
    Layers:   3 blocks, 4 heads, d=64
    Params:   151626
    Best ADE: 7.70m on held-out scenes
    Output:   learned/models/waypoint_transformer.pt
    Results:  learned/results/waypoint_transformer_eval.json

### nuScenes BEV Loader (Akilan Manivannan)

    python acquisition/nuscenes_bev.py

    Dataset:  nuScenes mini v1.0
    Scenes:   10 real driving scenes
    Samples:  404 keyframes
    Annotations: 18538 real 3D bounding boxes
    Ego poses:   31206 real vehicle positions

---

## Evaluation Gates

| Task | Metric | Gate | Result | Status |
|------|--------|------|--------|--------|
| Task B drowsiness | AUC | >= 0.90 | 0.9514 | PASS |
| Task A arrhythmia | Real data | PTB-XL | Real | PASS |
| Waypoint prediction | ADE | < 10.0m | 7.70m | PASS |
| SQI gating | Abstain | honest | Active | PASS |
| Medical grade | Guardrail | False | False | PASS |

---

## Signal Quality Gating

Guardian Drive abstains from inference when signal quality is insufficient.

    ECG quality < 0.5    Task A abstains
    EDA contact < 0.3    EDA contribution zeroed in Task B
    Motion level > 0.8   Crash detector priority raised
    Overall SQI < 0.4    Full INACTIVE state triggered

This prevents false positives from artifact-corrupted signals.
Designed by Akila Lourdes Miriyala Francis.

---

## Claim Guardrail

Located in policy/fusion.py:

    claim_guardrail = {
        "full_system_real":      False,   # honest
        "task_a_real_training":  True,    # PTB-XL verified
        "task_b_runtime_live":   True,    # WESAD verified
        "task_c_fully_real":     False,   # rule-based threshold
        "medical_grade":         False,   # not clinically validated
    }

---

## Inference Latency Budget

| Component | Latency | Owner |
|-----------|---------|-------|
| SQI computation | ~5ms | Akila |
| Feature extraction | ~10ms | Akila |
| Task A inference | ~15ms | Akila |
| Task B TCN | ~20ms | Akila |
| Task C IMU | ~2ms | Akilan |
| nuScenes BEV frame | ~8ms | Akilan |
| Policy state machine | ~1ms | Both |
| WebSocket broadcast | ~1ms | Akila |
| Voice alert (async) | non-blocking | Akilan |
| Total pipeline | ~50ms | Both |

---

## Experiment Tracking

Results stored in learned/results/:

    task_b_eval.json
      {"auc": 0.9514, "epochs_trained": 11}

    waypoint_transformer_eval.json
      {"ade_m": 7.70, "fde_m": 13.85, "params": 151626, "dataset": "nuScenes mini"}

---

## Reproducibility

    # Task B — fixed seed via PyTorch DataLoader
    python learned/task_b_trainer.py --data_dir datasets/WESAD/WESAD

    # Waypoint transformer
    python learned/waypoint_transformer.py --train

    # Verify inference
    python learned/waypoint_transformer.py --infer
    python acquisition/nuscenes_bev.py

---

## Model Versioning

    learned/models/
    |-- task_b_tcn.pt              v1.0  AUC 0.9514
    |-- waypoint_transformer.pt    v1.0  ADE 7.70m

    artifacts/
    |-- task_a_model_real_ptbxl.joblib   PTB-XL trained
    |-- drowsiness_cnn_subjectsplit.pt   CNN baseline

---

## Failure Modes & Mitigations

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| Noisy ECG | SQI < 0.5 | Task A abstains |
| No face detected | MediaPipe miss | Camera score = 0 |
| GPS unavailable | No fix | POI skipped gracefully |
| OSM timeout | 3s timeout | Route omitted |
| Discord fail | Exception catch | Silent, non-blocking |
| Pipeline exception | Try/except loop | Server continues |

---

## CI / Deployment

    # Start server
    GD_LIVE_ONLY=0 GD_ENABLE_WEBCAM=1 GD_WINDOW_SEC=8 GD_STEP_SEC=2 \
    GD_TASK_B_MLP_WEIGHTS=learned/models/task_b_tcn.pt GD_VOICE=1 \
    python -m uvicorn server.app:app --host 127.0.0.1 --port 8000

    # Health check
    curl http://127.0.0.1:8000/healthz

    # Runtime status
    curl http://127.0.0.1:8000/api/runtime_status

---

*Guardian Drive MLOps — Akila Lourdes Miriyala Francis & Akilan Manivannan — 2026*
