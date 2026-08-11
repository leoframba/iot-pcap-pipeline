"""Binary classification metrics for Phase 2B.2 baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from iot_pcap_pipeline.modeling.baselines.constants import DECISION_THRESHOLD, LABEL_MAPPING


@dataclass
class ConfusionCounts:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        yt = y_true.astype(np.uint8, copy=False)
        yp = y_pred.astype(np.uint8, copy=False)
        self.tp += int(np.sum((yt == 1) & (yp == 1)))
        self.fp += int(np.sum((yt == 0) & (yp == 1)))
        self.tn += int(np.sum((yt == 0) & (yp == 0)))
        self.fn += int(np.sum((yt == 1) & (yp == 0)))

    @property
    def support_positive(self) -> int:
        return self.tp + self.fn

    @property
    def support_negative(self) -> int:
        return self.tn + self.fp

    def as_dict(self) -> dict[str, int]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
        }


def _safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return float(num) / float(den)


def metrics_from_confusion(
    counts: ConfusionCounts,
    *,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, Any]:
    tp, fp, tn, fn = counts.tp, counts.fp, counts.tn, counts.fn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)  # attack recall
    specificity = _safe_div(tn, tn + fp)
    fpr = _safe_div(fp, fp + tn)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    bal = None
    if recall is not None and specificity is not None:
        bal = 0.5 * (recall + specificity)
    total = tp + fp + tn + fn
    accuracy = _safe_div(tp + tn, total)
    fp_per_10k = None if fpr is None else fpr * 10_000.0
    return {
        "threshold": threshold,
        "precision": precision,
        "attack_recall": recall,
        "f1": f1,
        "specificity": specificity,
        "benign_fpr": fpr,
        "benign_fp_count": fp,
        "benign_support": tn + fp,
        "false_positives_per_10k_benign": fp_per_10k,
        "balanced_accuracy": bal,
        "accuracy": accuracy,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def score_percentiles(scores: np.ndarray, ps: list[float]) -> dict[str, float | None]:
    if scores.size == 0:
        return {f"p{int(p):02d}" if p >= 1 else f"p{p}": None for p in ps}
    # Map requested percentiles to keys
    out: dict[str, float | None] = {}
    values = np.percentile(scores.astype(np.float64), ps)
    for p, v in zip(ps, values, strict=True):
        key = f"p{int(p):02d}" if float(p).is_integer() else f"p{p}"
        # Prefer explicit names used by the contract docs.
        if p == 5:
            key = "p05"
        elif p == 50:
            key = "p50"
        elif p == 95:
            key = "p95"
        elif p == 99:
            key = "p99"
        out[key] = float(v)
    return out


@dataclass
class RunningScoreStats:
    count: int = 0
    sum_score: float = 0.0
    _chunks: list[np.ndarray] = field(default_factory=list)

    def update(self, scores: np.ndarray) -> None:
        if scores.size == 0:
            return
        chunk = np.asarray(scores, dtype=np.float32)
        self.count += int(chunk.size)
        self.sum_score += float(np.sum(chunk, dtype=np.float64))
        self._chunks.append(chunk)

    def _array(self) -> np.ndarray:
        if not self._chunks:
            return np.empty(0, dtype=np.float32)
        if len(self._chunks) == 1:
            return self._chunks[0]
        return np.concatenate(self._chunks)

    def summary(self) -> dict[str, Any]:
        arr = self._array()
        mean = (self.sum_score / self.count) if self.count else None
        pct = score_percentiles(arr, [5, 50, 95, 99])
        maximum = float(arr.max()) if arr.size else None
        return {
            "attack_score_mean": mean,
            "attack_score_p05": pct.get("p05"),
            "attack_score_p50": pct.get("p50"),
            "attack_score_p95": pct.get("p95"),
            "attack_score_p99": pct.get("p99"),
            "max_attack_score": maximum,
        }


@dataclass
class GroupAccumulator:
    key: str
    kind: str  # attack_group | benign_group | pcap
    binary_label: str
    pcap_ids: set[str] = field(default_factory=set)
    counts: ConfusionCounts = field(default_factory=ConfusionCounts)
    score_stats: RunningScoreStats = field(default_factory=RunningScoreStats)

    def update(
        self,
        *,
        pcap_id: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        scores: np.ndarray,
    ) -> None:
        self.pcap_ids.add(pcap_id)
        self.counts.update(y_true, y_pred)
        self.score_stats.update(scores)

    def row_count(self) -> int:
        c = self.counts
        return c.tp + c.fp + c.tn + c.fn

    def to_attack_row(self) -> dict[str, Any]:
        m = metrics_from_confusion(self.counts)
        s = self.score_stats.summary()
        return {
            "modeling_group_key": self.key,
            "pcap_count": len(self.pcap_ids),
            "row_count": self.row_count(),
            "tp": self.counts.tp,
            "fn": self.counts.fn,
            "recall": m["attack_recall"],
            "attack_score_mean": s["attack_score_mean"],
            "attack_score_p05": s["attack_score_p05"],
            "attack_score_p50": s["attack_score_p50"],
            "attack_score_p95": s["attack_score_p95"],
        }

    def to_benign_row(self) -> dict[str, Any]:
        m = metrics_from_confusion(self.counts)
        s = self.score_stats.summary()
        return {
            "benign_group": self.key,
            "pcap_count": len(self.pcap_ids),
            "row_count": self.row_count(),
            "tn": self.counts.tn,
            "fp": self.counts.fp,
            "fpr": m["benign_fpr"],
            "specificity": m["specificity"],
            "attack_score_mean": s["attack_score_mean"],
            "attack_score_p95": s["attack_score_p95"],
            "attack_score_p99": s["attack_score_p99"],
            "max_attack_score": s["max_attack_score"],
        }

    def to_pcap_row(
        self,
        *,
        modeling_group_key: str,
        binary_label: str,
        benign_category: str,
    ) -> dict[str, Any]:
        m = metrics_from_confusion(self.counts)
        s = self.score_stats.summary()
        return {
            "pcap_id": self.key,
            "modeling_group_key": modeling_group_key,
            "binary_label": binary_label,
            "benign_category": benign_category,
            "row_count": self.row_count(),
            "tp": self.counts.tp,
            "fp": self.counts.fp,
            "tn": self.counts.tn,
            "fn": self.counts.fn,
            "recall": m["attack_recall"],
            "fpr": m["benign_fpr"],
            "specificity": m["specificity"],
            "attack_score_mean": s["attack_score_mean"],
            "attack_score_p95": s["attack_score_p95"],
            "attack_score_p99": s["attack_score_p99"],
            "max_attack_score": s["max_attack_score"],
        }


def benign_group_key(benign_category: str, modeling_group_key: str) -> str | None:
    """Map validation benign metadata to reporting buckets."""
    cat = (benign_category or "").strip()
    if cat == "profiling_idle":
        return "profiling_idle"
    if "Owltron" in modeling_group_key or "owltron" in modeling_group_key.lower():
        if cat == "profiling_interaction":
            return "owltron_interaction"
        if cat == "profiling_power":
            return "owltron_power"
    return None


def global_ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    if y_true.size == 0:
        return {"roc_auc": None, "pr_auc": None}
    # sklearn requires both classes for ROC-AUC.
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": None, "pr_auc": None}
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
    }


def macro_mean(values: list[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


ATTACK_LABEL = LABEL_MAPPING["ATTACK"]
BENIGN_LABEL = LABEL_MAPPING["BENIGN"]
