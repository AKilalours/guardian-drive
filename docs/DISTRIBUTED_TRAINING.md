# Guardian Drive — Distributed Training Scale-Up

> Built by Akila Lourdes Miriyala Francis & Akilan Manivannan

## Overview

Guardian Drive's multi-task 4-head training system is architecturally
ready for PyTorch DistributedDataParallel (DDP) scaling. This document
describes the scaling path from single-device to multi-GPU training.

## Current Training (Single Device)

    # Task B — WESAD TCN (AUC 0.9514, Apple M4 CPU)
    python learned/task_b_trainer.py --data_dir datasets/WESAD/WESAD

    # Waypoint Transformer (nuScenes 31,206 demonstrations)
    python learned/waypoint_transformer.py --train

## DDP Scale-Up — PyTorch DistributedDataParallel

### Why DDP fits Guardian Drive

The multi-task 4-head fusion system (Task A + B + C + D) maps
directly to DDP because:
- Each task head is an independent nn.Module
- All heads share the same input feature batch
- Gradient sync across heads is handled automatically by DDP
- Batch size scales linearly with GPU count

### Launch Commands

```bash
# 2-GPU DDP training — Task B TCN
torchrun --nproc_per_node=2 learned/task_b_trainer.py \
    --data_dir datasets/WESAD/WESAD \
    --ddp

# 4-GPU DDP training — Waypoint Transformer
torchrun --nproc_per_node=4 learned/waypoint_transformer.py \
    --train --ddp

# Multi-node (2 nodes x 4 GPUs = 8 GPUs total)
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="master_node_ip" \
    --master_port=29500 \
    learned/task_b_trainer.py --data_dir datasets/WESAD/WESAD --ddp
```

### DDP Implementation Pattern

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def train_ddp(rank, world_size):
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Wrap model
    model = WESADTCNModel().to(rank)
    model = DDP(model, device_ids=[rank])

    # Scale batch size linearly with GPU count
    batch_size = 64 * world_size  # 64 per GPU

    # DistributedSampler ensures no data overlap across GPUs
    from torch.utils.data.distributed import DistributedSampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # Training loop — gradients sync automatically via all-reduce
    for batch in loader:
        optimizer.zero_grad()
        loss = criterion(model(batch))
        loss.backward()  # DDP handles gradient synchronization
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    dist.destroy_process_group()
```

### Scaling Properties

| GPUs | Batch Size | Expected Speedup | Memory per GPU |
|------|-----------|-----------------|----------------|
| 1 | 64 | 1x (baseline) | ~2 GB |
| 2 | 128 | ~1.9x | ~2 GB |
| 4 | 256 | ~3.7x | ~2 GB |
| 8 | 512 | ~7.2x | ~2 GB |

### Gradient Synchronization

DDP uses NCCL all-reduce for gradient synchronization:
- Ring all-reduce across all GPUs after each backward pass
- Communication overlaps with computation (bucketed gradients)
- Effective batch gradient = average across all GPU gradients
- Linear scaling rule: LR scales with batch size (LR x world_size)

### Multi-Task 4-Head DDP

```python
# All 4 task heads wrapped in single DDP call
class MultiTaskFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.task_a = ArrhythmiaHead()    # PTB-XL RandomForest + CNN
        self.task_b = DrowsinessHead()    # WESAD TCN AUC 0.9514
        self.task_c = CrashHead()         # IMU g-force
        self.task_d = NeuroRiskHead()     # HRV + EDA

    def forward(self, x):
        return {
            "task_a": self.task_a(x),
            "task_b": self.task_b(x),
            "task_c": self.task_c(x),
            "task_d": self.task_d(x),
        }

# Single DDP wrap covers all 4 heads simultaneously
model = DDP(MultiTaskFusion().to(rank), device_ids=[rank])
```

### Current Bottleneck (Mini Dataset)

nuScenes mini (404 samples) and WESAD (2,874 windows) are too small
to benefit from DDP — communication overhead would exceed compute time.
DDP becomes beneficial at:
- WESAD full dataset (>10,000 windows)
- nuScenes full (700+ scenes, ~390,000 ego poses)
- PTB-XL full (21,837 clinical ECG records)

## Resume Claim

> "Guardian Drive's multi-task 4-head architecture
> (Task A/B/C/D) is designed for PyTorch DDP scaling —
> each head is an independent nn.Module, batch size scales
> linearly with GPU count, and gradient synchronization
> is handled via NCCL all-reduce. Documented scale-up
> from 1 to 8 GPUs with linear batch scaling."
