"""TRAIN-validation-only evaluator for D0 PCAP aggregation candidates.

Scores frozen H0 window predictions on unsampled TRAIN-validation PCAPs,
applies the predeclared (K, R) grid, and writes selection evidence.
Does not freeze K/R and must never read TEST feature shards.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from iot_pcap_pipeline.modeling.baselines.data import (
    load_validation_specs,
    iter_validation_batches,
    validate_validation_inventory,
    reject_test_path,
)
from iot_pcap_pipeline.modeling.baselines.model_input import V1_MODEL_INPUT_FEATURES
from iot_pcap_pipeline.modeling.baselines.models import attack_score_from_estimator
from iot_pcap_pipeline.modeling.view import file_sha256
from iot_pcap_pipeline.paths import (
    DEFAULT_MODELING_DIR,
    PROJECT_ROOT,
    to_repo_relative,
)
from iot_pcap_pipeline.serving.contract import (
    FROZEN_ATTACK_RATE_THRESHOLD,
    FROZEN_MIN_ATTACK_WINDOWS,
    FROZEN_MIN_COMPLETE_WINDOWS,
)
from iot_pcap_pipeline.serving.candidates import (
    AggregationPolicy,
    DEFAULT_CANDIDATES_PATH,
    WINDOW_ATTACK_THRESHOLD,
    iter_candidate_policies,
    load_candidates_document,
    verify_candidates_document,
)
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError

DEFAULT_SERVING_DIR = PROJECT_ROOT / "data" / "serving" / "v1"
DEFAULT_SPLIT_MANIFEST = DEFAULT_MODELING_DIR / "v1" / "modeling_split_manifest.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "v1" / "H0_full_fit.joblib"
EXPECTED_MODEL_SHA256 = (
    "c07ef4088cd44523787c041db449f64429328c0a42b76dfe14de3697cbea77bb"
)
FEATURES_22 = list(V1_MODEL_INPUT_FEATURES)


@dataclass(frozen=True)
class PcapWindowTape:
    pcap_id: str
    pcap_path: str
    binary_label: str
    attack_family: str
    attack_type: str
    benign_category: str
    modeling_group_key: str
    total_windows: int
    attack_windows: int
    benign_windows: int
    pcap_attack_score: float | None
    max_window_attack_score: float | None
    mean_window_attack_score: float | None
    status: str  # OK | INSUFFICIENT_DATA


def _decide_pcap(
    *,
    total_windows: int,
    attack_windows: int,
    policy: AggregationPolicy,
    minimum_complete_windows: int | None = None,
) -> str:
    min_windows = (
        int(minimum_complete_windows)
        if minimum_complete_windows is not None
        else int(policy.K)
    )
    if total_windows < min_windows:
        return "INSUFFICIENT_DATA"
    rate = attack_windows / total_windows
    if attack_windows >= policy.K and rate >= policy.R:
        return "ATTACK"
    return "BENIGN"


def score_validation_pcaps(
    *,
    project_root: Path | None = None,
    split_manifest_path: Path | None = None,
    model_path: Path | None = None,
    expected_model_sha256: str = EXPECTED_MODEL_SHA256,
    window_threshold: float = WINDOW_ATTACK_THRESHOLD,
) -> list[PcapWindowTape]:
    """Score each TRAIN-validation PCAP with frozen H0; refuse TEST paths."""
    root = (project_root or PROJECT_ROOT).resolve()
    if window_threshold != WINDOW_ATTACK_THRESHOLD:
        raise FeatureExtractionError(
            f"window threshold override rejected: {window_threshold!r}"
        )

    split_path = Path(split_manifest_path or DEFAULT_SPLIT_MANIFEST)
    if not split_path.is_absolute():
        split_path = root / split_path
    reject_test_path(split_path)

    model_file = Path(model_path or DEFAULT_MODEL_PATH)
    if not model_file.is_absolute():
        model_file = root / model_file
    digest = file_sha256(model_file)
    if digest != expected_model_sha256:
        raise FeatureExtractionError(
            f"model SHA mismatch: actual={digest} expected={expected_model_sha256}"
        )
    estimator = joblib.load(model_file)

    specs = load_validation_specs(split_path, project_root=root)
    validate_validation_inventory(specs, smoke_only=False)

    # Accumulators keyed by pcap_id (specs are unique).
    totals = {s.pcap_id: 0 for s in specs}
    attacks = {s.pcap_id: 0 for s in specs}
    score_sum = {s.pcap_id: 0.0 for s in specs}
    score_max = {s.pcap_id: float("-inf") for s in specs}
    meta = {s.pcap_id: s for s in specs}

    for batch in iter_validation_batches(
        specs,
        project_root=root,
        feature_names=FEATURES_22,
    ):
        reject_test_path(batch.spec.feature_parquet_path)
        scores = np.asarray(
            attack_score_from_estimator(estimator, batch.X), dtype=np.float64
        )
        if scores.shape[0] != batch.X.shape[0]:
            raise FeatureExtractionError("score length mismatch vs feature rows")
        pred_attack = scores >= window_threshold
        pid = batch.spec.pcap_id
        n = int(scores.shape[0])
        totals[pid] += n
        attacks[pid] += int(pred_attack.sum())
        score_sum[pid] += float(scores.sum())
        if n:
            score_max[pid] = max(score_max[pid], float(scores.max()))

    tapes: list[PcapWindowTape] = []
    for spec in sorted(specs, key=lambda s: s.pcap_id):
        total = int(totals[spec.pcap_id])
        atk = int(attacks[spec.pcap_id])
        if total == 0:
            tapes.append(
                PcapWindowTape(
                    pcap_id=spec.pcap_id,
                    pcap_path=spec.pcap_path,
                    binary_label=spec.binary_label,
                    attack_family=spec.attack_family,
                    attack_type=spec.attack_type,
                    benign_category=spec.benign_category,
                    modeling_group_key=spec.modeling_group_key,
                    total_windows=0,
                    attack_windows=0,
                    benign_windows=0,
                    pcap_attack_score=None,
                    max_window_attack_score=None,
                    mean_window_attack_score=None,
                    status="INSUFFICIENT_DATA",
                )
            )
            continue
        if total != int(spec.window_count):
            raise FeatureExtractionError(
                f"window count drift for {spec.pcap_id}: "
                f"scored={total} manifest={spec.window_count}"
            )
        if total < FROZEN_MIN_COMPLETE_WINDOWS:
            tapes.append(
                PcapWindowTape(
                    pcap_id=spec.pcap_id,
                    pcap_path=spec.pcap_path,
                    binary_label=spec.binary_label,
                    attack_family=spec.attack_family,
                    attack_type=spec.attack_type,
                    benign_category=spec.benign_category,
                    modeling_group_key=spec.modeling_group_key,
                    total_windows=total,
                    attack_windows=atk,
                    benign_windows=total - atk,
                    pcap_attack_score=None,
                    max_window_attack_score=(
                        None
                        if score_max[spec.pcap_id] == float("-inf")
                        else score_max[spec.pcap_id]
                    ),
                    mean_window_attack_score=(
                        None if total == 0 else score_sum[spec.pcap_id] / total
                    ),
                    status="INSUFFICIENT_DATA",
                )
            )
            continue
        tapes.append(
            PcapWindowTape(
                pcap_id=spec.pcap_id,
                pcap_path=spec.pcap_path,
                binary_label=spec.binary_label,
                attack_family=spec.attack_family,
                attack_type=spec.attack_type,
                benign_category=spec.benign_category,
                modeling_group_key=spec.modeling_group_key,
                total_windows=total,
                attack_windows=atk,
                benign_windows=total - atk,
                pcap_attack_score=atk / total,
                max_window_attack_score=(
                    None if score_max[spec.pcap_id] == float("-inf") else score_max[spec.pcap_id]
                ),
                mean_window_attack_score=score_sum[spec.pcap_id] / total,
                status="OK",
            )
        )
    return tapes


def summarize_policies(
    tapes: list[PcapWindowTape],
    policies: tuple[AggregationPolicy, ...] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (by_pcap rows, policy summary rows)."""
    pols = policies or iter_candidate_policies()
    by_pcap: list[dict[str, Any]] = []
    for tape in tapes:
        row: dict[str, Any] = {
            "pcap_id": tape.pcap_id,
            "pcap_path": tape.pcap_path,
            "binary_label": tape.binary_label,
            "attack_family": tape.attack_family,
            "attack_type": tape.attack_type,
            "benign_category": tape.benign_category,
            "modeling_group_key": tape.modeling_group_key,
            "status": tape.status,
            "total_windows": tape.total_windows,
            "attack_windows": tape.attack_windows,
            "benign_windows": tape.benign_windows,
            "pcap_attack_score": (
                "" if tape.pcap_attack_score is None else f"{tape.pcap_attack_score:.10f}"
            ),
            "max_window_attack_score": (
                ""
                if tape.max_window_attack_score is None
                else f"{tape.max_window_attack_score:.10f}"
            ),
            "mean_window_attack_score": (
                ""
                if tape.mean_window_attack_score is None
                else f"{tape.mean_window_attack_score:.10f}"
            ),
        }
        for pol in pols:
            row[pol.policy_id] = _decide_pcap(
                total_windows=tape.total_windows,
                attack_windows=tape.attack_windows,
                policy=pol,
            )
        by_pcap.append(row)

    summaries: list[dict[str, Any]] = []
    attack_tapes = [t for t in tapes if t.binary_label == "ATTACK"]
    benign_tapes = [t for t in tapes if t.binary_label == "BENIGN"]
    families = sorted(
        {t.attack_family for t in attack_tapes if t.attack_family}
    )

    for pol in pols:
        benign_fp = 0
        benign_scored = 0
        for t in benign_tapes:
            pred = _decide_pcap(
                total_windows=t.total_windows,
                attack_windows=t.attack_windows,
                policy=pol,
            )
            if pred == "INSUFFICIENT_DATA":
                continue
            benign_scored += 1
            if pred == "ATTACK":
                benign_fp += 1

        attack_tp = 0
        attack_scored = 0
        for t in attack_tapes:
            pred = _decide_pcap(
                total_windows=t.total_windows,
                attack_windows=t.attack_windows,
                policy=pol,
            )
            if pred == "INSUFFICIENT_DATA":
                continue
            attack_scored += 1
            if pred == "ATTACK":
                attack_tp += 1

        family_recalls: dict[str, float] = {}
        for fam in families:
            fam_tapes = [t for t in attack_tapes if t.attack_family == fam]
            fam_scored = 0
            fam_tp = 0
            for t in fam_tapes:
                pred = _decide_pcap(
                    total_windows=t.total_windows,
                    attack_windows=t.attack_windows,
                    policy=pol,
                )
                if pred == "INSUFFICIENT_DATA":
                    continue
                fam_scored += 1
                if pred == "ATTACK":
                    fam_tp += 1
            family_recalls[fam] = (fam_tp / fam_scored) if fam_scored else float("nan")

        macro = (
            float(np.nanmean(list(family_recalls.values())))
            if family_recalls
            else float("nan")
        )
        min_fam = (
            float(np.nanmin(list(family_recalls.values())))
            if family_recalls
            else float("nan")
        )
        summaries.append(
            {
                "policy_id": pol.policy_id,
                "K": pol.K,
                "R": pol.R,
                "benign_pcap_scored": benign_scored,
                "benign_pcap_false_positives": benign_fp,
                "benign_pcap_fpr": (
                    benign_fp / benign_scored if benign_scored else float("nan")
                ),
                "attack_pcap_scored": attack_scored,
                "attack_pcap_true_positives": attack_tp,
                "attack_pcap_recall": (
                    attack_tp / attack_scored if attack_scored else float("nan")
                ),
                "macro_family_recall": macro,
                "min_family_recall": min_fam,
                **{f"family_recall_{fam}": family_recalls[fam] for fam in families},
            }
        )
    return by_pcap, summaries


def rank_policies(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply predeclared selection priority; does not freeze."""

    def sort_key(row: dict[str, Any]) -> tuple:
        return (
            int(row["benign_pcap_false_positives"]),
            -float(row["attack_pcap_recall"]),
            -float(row["macro_family_recall"]),
            int(row["K"]),
            float(row["R"]),
        )

    ranked = sorted(summaries, key=sort_key)
    out: list[dict[str, Any]] = []
    for i, row in enumerate(ranked, start=1):
        item = dict(row)
        item["rank"] = i
        item["recommended"] = i == 1
        item["freeze_status"] = "pending_human_review"
        out.append(item)
    return out


def write_aggregation_evaluation(
    *,
    project_root: Path | None = None,
    out_dir: Path | None = None,
    split_manifest_path: Path | None = None,
    model_path: Path | None = None,
) -> dict[str, Any]:
    """Run VAL-only evaluation and write selection artifacts."""
    root = (project_root or PROJECT_ROOT).resolve()
    out = Path(out_dir or DEFAULT_SERVING_DIR)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates_document(project_root=root)
    verify_candidates_document(candidates)
    policies = iter_candidate_policies()

    tapes = score_validation_pcaps(
        project_root=root,
        split_manifest_path=split_manifest_path,
        model_path=model_path,
    )
    by_pcap, summaries = summarize_policies(tapes, policies)
    ranked = rank_policies(summaries)

    by_pcap_path = out / "pcap_aggregation_by_pcap.csv"
    summary_path = out / "pcap_aggregation_summary.csv"
    review_path = out / "pcap_aggregation_review.json"

    if by_pcap:
        fieldnames = list(by_pcap[0].keys())
        with by_pcap_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(by_pcap)

    if ranked:
        fieldnames = list(ranked[0].keys())
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in ranked:
                out_row = {
                    k: (
                        f"{v:.10f}"
                        if isinstance(v, float)
                        else v
                    )
                    for k, v in row.items()
                }
                writer.writerow(out_row)

    recommended = ranked[0] if ranked else None
    payload = {
        "status": "candidates_evaluated_pending_freeze",
        "selection_split": "TRAIN-validation",
        "window_attack_threshold": WINDOW_ATTACK_THRESHOLD,
        "model_artifact": to_repo_relative(DEFAULT_MODEL_PATH, project_root=root),
        "model_artifact_sha256": EXPECTED_MODEL_SHA256,
        "n_validation_pcaps": len(tapes),
        "n_attack_pcaps": sum(1 for t in tapes if t.binary_label == "ATTACK"),
        "n_benign_pcaps": sum(1 for t in tapes if t.binary_label == "BENIGN"),
        "n_insufficient_data": sum(
            1 for t in tapes if t.status == "INSUFFICIENT_DATA"
        ),
        "policies_evaluated": len(policies),
        "selection_priority": candidates.get("selection_priority"),
        "recommended_policy": (
            {
                "policy_id": recommended["policy_id"],
                "K": recommended["K"],
                "R": recommended["R"],
                "benign_pcap_false_positives": recommended[
                    "benign_pcap_false_positives"
                ],
                "attack_pcap_recall": recommended["attack_pcap_recall"],
                "macro_family_recall": recommended["macro_family_recall"],
                "min_family_recall": recommended["min_family_recall"],
                "note": "Recommendation only. Do not freeze without human review.",
            }
            if recommended
            else None
        ),
        "artifacts": {
            "candidates": to_repo_relative(DEFAULT_CANDIDATES_PATH, project_root=root),
            "by_pcap": to_repo_relative(by_pcap_path, project_root=root),
            "summary": to_repo_relative(summary_path, project_root=root),
            "review": to_repo_relative(review_path, project_root=root),
        },
        "next": (
            "Review pcap_aggregation_summary.csv. Freeze K/R into "
            "artifacts/v1/serving_contract.json only after explicit approval. "
            "Do not use TEST to choose K/R."
        ),
    }
    review_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def format_review_summary(payload: dict[str, Any]) -> str:
    rec = payload.get("recommended_policy") or {}
    lines = [
        "D0 PCAP aggregation — TRAIN-validation candidate review",
        f"validation_pcaps: {payload.get('n_validation_pcaps')}",
        f"attack/benign/insufficient: "
        f"{payload.get('n_attack_pcaps')}/"
        f"{payload.get('n_benign_pcaps')}/"
        f"{payload.get('n_insufficient_data')}",
        f"recommended (pending freeze): {rec.get('policy_id')} "
        f"(K={rec.get('K')}, R={rec.get('R')})",
        f"benign_pcap_fp={rec.get('benign_pcap_false_positives')} "
        f"attack_recall={rec.get('attack_pcap_recall')} "
        f"macro_family_recall={rec.get('macro_family_recall')}",
        f"artifacts: {payload.get('artifacts')}",
        "STOP: human review required before freezing K/R.",
    ]
    return "\n".join(lines)
