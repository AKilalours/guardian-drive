# Guardian Drive: real-data cleanup and training reality check

## What is duplicated right now

You have the same PTBDB content in three places:
- `data/raw/ptbdb/1.0.0`
- `datasets/ptbdb/1.0.0`
- `datasets/physionet.org/files/ptbdb/1.0.0`

You have the same WESAD content in two places:
- `data/raw/WESAD/WESAD`
- `datasets/WESAD/WESAD`

You also still have extracted/unextracted archives:
- `datasets/WESAD.zip`
- `datasets/sleep-edfx.zip`

And you have an unrelated stroke dataset:
- `datasets/isles2022`

## What to keep

Keep one canonical raw-data root:
- `data/raw/ptbdb/1.0.0`
- `data/raw/WESAD/WESAD`

Everything else should either be:
- archived, or
- replaced with symlinks back to `data/raw/...`

Do **not** flatten all datasets into one directory. That destroys structure and makes loaders brittle.

## What you can honestly claim after cleanup

After cleanup alone, you can claim only this:
- Guardian Drive uses real datasets stored in a canonical raw-data layout.

You still cannot honestly claim all tasks are real, because the code you uploaded says otherwise.

## What the code actually says

### Task A training is synthetic
`models/train_task_a.py` literally says:
- `Train Task A (Arrhythmia) model on *simulated* data.`
- `note: Trained on SIMULATED data only`

### Task C training is synthetic
`models/train_task_c.py` literally says:
- `Train Task C (Crash) model on *simulated* data.`
- `note: Trained on SIMULATED data only`

### Task B real training is not here
`models/train_task_b.py` is just a wrapper to:
- `learned.train_task_b`

So the real Task B training implementation is not in the files you uploaded.

## Brutal truth about your available datasets

### Task A — Arrhythmia
You do have real ECG data for this:
- PTBDB

This is the one task you can push toward a credible real-data training claim first.

### Task B — Drowsiness
You do **not** have a real driver-drowsiness dataset here.
- WESAD is not a driver-camera drowsiness dataset.
- Sleep-EDFX is sleep data, not in-vehicle drowsiness monitoring.

At best, those are weak physiological proxies. Calling that “real driver drowsiness” is misleading.

### Task C — Crash
You do **not** have a real crash dataset in what you showed.
So Task C cannot be honestly called real yet.

## Evaluation issue you still need to fix

In `evaluation/runner.py`, Task C currently sets:
- `score=float(np.clip(rs.crash.confidence, 0.0, 1.0))`

That only works if `rs.crash.confidence` is truly `P(crash)`.
But in `models/task_c.py`, for non-crash predictions it sets:
- `confidence = 1.0 - proba`

So your evaluation score semantics are inconsistent unless you convert back to positive-class probability. That is exactly why your AUC/ECE looked suspicious.

## Immediate sequence

### 1) Deduplicate safely
Run:
```bash
bash /mnt/data/gd_dedup_unify.sh
```
Review the diffs.
Then apply:
```bash
bash /mnt/data/gd_dedup_unify.sh --apply
```

### 2) Use the correct WESAD root
```bash
export GD_WESAD_ROOT="$PWD/data/raw/WESAD/WESAD"
export GD_WESAD_SUBJECT="S2"
```

### 3) Stop saying Task A/C are real until the training code is replaced
Right now the training scripts themselves contradict that claim.

### 4) Build real-data training in this order
1. Task A from PTBDB
2. Task B from an actual driver-drowsiness dataset or your own webcam collection
3. Task C from an actual crash/near-crash inertial dataset

## What I would do next

- clean storage first
- patch evaluation score semantics for Task C
- replace synthetic Task A training with PTBDB training
- do not pretend WESAD solves driver drowsiness
- do not pretend Task C is real without a real crash dataset
