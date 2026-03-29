"""Guardian Drive v4.1 - Evaluation Metrics

Metrics emphasized for safety-oriented systems:
- FAR/hour (false alarms per driving hour)
- Sensitivity/recall, Specificity
- ROC-AUC / PR-AUC from probability scores (not raw confidences)
- Expected Calibration Error (ECE) on probability scores
- Abstain rate

Important:
- If a record provides `score`, it is treated as P(positive_label|window).
- Otherwise we derive a score from (pred_label, confidence) assuming confidence is
  P(pred_label correct). This is a fallback for rule-based baselines.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:  # pragma: no cover
    roc_auc_score = None
    average_precision_score = None


@dataclass
class DetectionRecord:
    subject_id: str
    session_id: str
    window_start: float
    true_label: str
    pred_label: str
    confidence: float
    abstained: bool
    score: Optional[float] = None  # P(positive_label) for AUC/ECE
    latency_ms: float = 0.0        # Task C only


@dataclass
class EvaluationReport:
    task: str = ""
    n_windows: int = 0
    n_subjects: int = 0
    abstain_rate: float = 0.0
    sensitivity: float = 0.0
    specificity: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    far_per_hour: float = 0.0
    ece: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    notes: str = ""

    def summary(self) -> str:
        lines = [
            f"Task: {self.task}",
            f"Windows: {self.n_windows}  Subjects: {self.n_subjects}  Abstain: {self.abstain_rate:.1%}",
            f"Sensitivity: {self.sensitivity:.3f}  Specificity: {self.specificity:.3f}  Precision: {self.precision:.3f}",
            f"F1: {self.f1:.3f}  ROC-AUC: {self.roc_auc:.3f}  PR-AUC: {self.pr_auc:.3f}",
            f"FAR/hour: {self.far_per_hour:.2f}  ECE: {self.ece:.3f}",
        ]
        if self.p50_latency_ms > 0:
            lines.append(f"Latency p50: {self.p50_latency_ms:.0f}ms  p95: {self.p95_latency_ms:.0f}ms")
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        return "\n".join(lines)


def compute_far_per_hour(records: list[DetectionRecord], window_sec: float, positive_label: str = "arrhythmia") -> float:
    neg = [r for r in records if (not r.abstained) and (r.true_label != positive_label)]
    fp = sum(1 for r in neg if r.pred_label == positive_label)
    hours = len(neg) * window_sec / 3600.0
    return float(fp / hours) if hours > 0 else 0.0


def _score_positive(r: DetectionRecord, positive_label: str) -> float:
    if r.score is not None:
        return float(np.clip(r.score, 0.0, 1.0))
    # Fallback: assume confidence is P(pred_label correct)
    c = float(np.clip(r.confidence, 0.0, 1.0))
    return c if r.pred_label == positive_label else (1.0 - c)


def compute_ece(records: list[DetectionRecord], positive_label: str = "arrhythmia", n_bins: int = 10) -> float:
    valid = [r for r in records if not r.abstained]
    if not valid:
        return 0.0
    scores = np.array([_score_positive(r, positive_label) for r in valid], dtype=float)
    y_true = np.array([1 if r.true_label == positive_label else 0 for r in valid], dtype=int)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (scores >= bins[i]) & (scores < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc_b = float(y_true[mask].mean())
        conf_b = float(scores[mask].mean())
        ece += (mask.sum() / len(scores)) * abs(acc_b - conf_b)
    return float(ece)


def compute_metrics_binary(records: list[DetectionRecord], positive_label: str, window_sec: float) -> dict:
    valid = [r for r in records if not r.abstained]
    if not valid:
        return dict(sensitivity=0.0, specificity=0.0, precision=0.0, f1=0.0, roc_auc=0.0, pr_auc=0.0, far_per_hour=0.0)

    y_true = np.array([1 if r.true_label == positive_label else 0 for r in valid], dtype=int)
    y_pred = np.array([1 if r.pred_label == positive_label else 0 for r in valid], dtype=int)
    y_score = np.array([_score_positive(r, positive_label) for r in valid], dtype=float)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    f1 = 2 * prec * sens / (prec + sens + 1e-9)

    roc_auc = 0.0
    pr_auc = 0.0
    if roc_auc_score is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            roc_auc = 0.0
    if average_precision_score is not None:
        try:
            pr_auc = float(average_precision_score(y_true, y_score))
        except Exception:
            pr_auc = 0.0

    far = compute_far_per_hour(records, window_sec, positive_label)

    return dict(
        sensitivity=float(sens),
        specificity=float(spec),
        precision=float(prec),
        f1=float(f1),
        roc_auc=float(np.clip(roc_auc, 0.0, 1.0)),
        pr_auc=float(np.clip(pr_auc, 0.0, 1.0)),
        far_per_hour=float(far),
    )


def loso_evaluation(all_records: list[DetectionRecord], positive_label: str, task_name: str, window_sec: float, notes: str = "") -> EvaluationReport:
    subjects = sorted(set(r.subject_id for r in all_records))
    fold = []
    for subj in subjects:
        test = [r for r in all_records if r.subject_id == subj]
        fold.append(compute_metrics_binary(test, positive_label, window_sec))

    avg = {k: float(np.mean([fm[k] for fm in fold])) for k in fold[0]}
    abstain_rate = float(sum(1 for r in all_records if r.abstained) / max(1, len(all_records)))
    latencies = [r.latency_ms for r in all_records if r.latency_ms > 0]

    rep = EvaluationReport(
        task=task_name,
        n_windows=len(all_records),
        n_subjects=len(subjects),
        abstain_rate=abstain_rate,
        ece=compute_ece(all_records, positive_label),
        p50_latency_ms=float(np.percentile(latencies, 50)) if latencies else 0.0,
        p95_latency_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
        notes=notes,
        **avg,
    )
    return rep
