<p align="center">
  <img src="Cover_image.png" alt="Guardian Drive Cover" width="900">
</p>

# GuardianDrive — Physio Baseline (WESAD) with SQI Gating + LOSO Eval

Phase 1 implements:
- WESAD chest physiology pipeline (ECG/EDA/RESP/Temp)
- Windowing + train-only normalization (no leakage)
- LOSO-style split via subject holdout (train / val / test subjects)
- Signal Quality Index (SQI) gating + abstain behavior
- Robustness checks (noise + channel dropout)
- Latency benchmark (p50/p95)

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
make dev
```
now im working on this project, in this i have used the LLM,RAF,Agentic AIand Gen AI

