# Guardian Drive -- MLOps & Model Lifecycle

## Model Registry

| Model | Path | Framework | Result | Training Data |
|-------|------|-----------|--------|---------------|
| Task B TCN | learned/models/task_b_tcn.pt | PyTorch | AUC 0.9514 | WESAD 15 subjects |
| Task A RF | artifacts/task_a_model_real_ptbxl.joblib | Scikit-learn | Real PTB-XL | Clinical ECG |
| Waypoint Transformer | learned/models/waypoint_transformer.pt | PyTorch | ADE 7.70m | nuScenes 31,206 poses |
| Drowsiness CNN | artifacts/drowsiness_cnn_subjectsplit.pt | PyTorch | Baseline | Subject-split |

## Training Commands

### Task B -- WESAD TCN
    python learned/task_b_trainer.py --data_dir datasets/WESAD/WESAD
    # Output: learned/models/task_b_tcn.pt
    # Result: learned/results/task_b_eval.json -- AUC 0.9514

### Task A -- PTB-XL
    pip install wfdb
    python learned/task_a_trainer.py --data_dir datasets/ptbdb/1.0.0
    # Output: artifacts/task_a_model_real_ptbxl.joblib

### Waypoint Transformer
    python learned/waypoint_transformer.py --train
    # Output: learned/models/waypoint_transformer.pt
    # Result: learned/results/waypoint_transformer_eval.json -- ADE 7.70m

## Evaluation Gates

| Task | Metric | Gate | Result |
|------|--------|------|--------|
| Task B drowsiness | AUC | >= 0.90 | 0.9514 PASS |
| Task A arrhythmia | Real data | PTB-XL | PASS |
| Waypoint prediction | ADE | < 10.0m | 7.70m PASS |
| SQI gating | Abstain | honest | PASS |

## Signal Quality Gating

Guardian Drive abstains from inference when signal quality is insufficient.
ECG quality < 0.5 causes Task A to abstain.
EDA contact < 0.3 zeros the EDA contribution to Task B.
Motion level > 0.8 raises crash detector priority.
Overall SQI < 0.4 triggers full INACTIVE state.

## Claim Guardrail

    claim_guardrail = {
        "full_system_real": False,
        "task_a_real_training": True,
        "task_b_runtime_live": True,
        "task_c_fully_real": False,
        "medical_grade": False,
    }

## Inference Latency

| Component | Latency |
|-----------|---------|
| SQI computation | ~5ms |
| Feature extraction | ~10ms |
| Task A inference | ~15ms |
| Task B TCN | ~20ms |
| Policy state machine | ~1ms |
| WebSocket broadcast | ~1ms |
| Total pipeline | ~50ms |
