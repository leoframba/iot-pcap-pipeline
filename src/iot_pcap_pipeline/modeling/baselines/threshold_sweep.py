"""Phase 2B.3A: threshold sweep on frozen 2B.2 models (no retraining)."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import joblib
import numpy as np

from iot_pcap_pipeline.modeling.baselines.constants import (
    ATTACK_VAL_GROUPS,
    BASELINE_STRATEGY_VERSION,
    EXPECTED_VAL_ROWS,
    LABEL_MAPPING,
)
from iot_pcap_pipeline.modeling.baselines.contract import (
    DEFAULT_BASELINES_ROOT,
    load_frozen_baseline_contract,
)
from iot_pcap_pipeline.modeling.baselines.data import (
    load_validation_specs,
    iter_validation_batches,
    validate_validation_inventory,
)
from iot_pcap_pipeline.modeling.baselines.metrics import benign_group_key
from iot_pcap_pipeline.modeling.baselines.models import (
    MODEL_SPECS,
    attack_score_from_estimator,
)
from iot_pcap_pipeline.modeling.view import (
    DEFAULT_SPLIT_MANIFEST_PATH,
    file_sha256,
)
from iot_pcap_pipeline.paths import (
    DEFAULT_MODELING_DIR,
    PROJECT_ROOT,
    to_repo_relative,
)
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError

THRESHOLD_SWEEP_VERSION = "phase2b3a_v1"
DEFAULT_SWEEP_ROOT = (
    DEFAULT_MODELING_DIR / "v1" / "baselines" / THRESHOLD_SWEEP_VERSION
)

FIXED_THRESHOLDS: tuple[float, ...] = (
    0.50,
    0.75,
    0.85,
    0.90,
    0.925,
    0.94,
    0.95,
    0.97,
    0.98,
    0.99,
    0.995,
)

BENIGN_FPR_TARGETS: tuple[float, ...] = (
    0.10,
    0.05,
    0.02,
    0.01,
    0.005,
    0.001,
)

# Compact group codes for the validation score tape.
ATTACK_GROUP_CODES: dict[str, int] = {
    "DDoS|DDoS_TCP": 1,
    "DoS|DoS_TCP": 2,
    "MQTT|MQTT_DoS_Publish_Flood": 3,
    "Recon|OS_Scan": 4,
}
BENIGN_GROUP_CODES: dict[str, int] = {
    "profiling_idle": 10,
    "owltron_interaction": 11,
    "owltron_power": 12,
}
CODE_TO_GROUP: dict[int, str] = {
    **{v: k for k, v in ATTACK_GROUP_CODES.items()},
    **{v: k for k, v in BENIGN_GROUP_CODES.items()},
}

SWEEP_ROW_COLUMNS: tuple[str, ...] = (
    "model_id",
    "point_type",  # fixed_threshold | fpr_target
    "threshold",
    "fpr_target",
    "benign_fp",
    "benign_support",
    "benign_fpr",
    "owltron_interaction_fp",
    "owltron_interaction_support",
    "owltron_interaction_fpr",
    "profiling_idle_fp",
    "profiling_idle_support",
    "profiling_idle_fpr",
    "owltron_power_fp",
    "owltron_power_support",
    "owltron_power_fpr",
    "ddos_tcp_tp",
    "ddos_tcp_support",
    "ddos_tcp_recall",
    "dos_tcp_tp",
    "dos_tcp_support",
    "dos_tcp_recall",
    "mqtt_publish_tp",
    "mqtt_publish_support",
    "mqtt_publish_recall",
    "recon_os_scan_tp",
    "recon_os_scan_support",
    "recon_os_scan_recall",
    "macro_attack_group_recall",
    "min_attack_group_recall",
)


@dataclass
class ValidationScoreTape:
    y_true: np.ndarray
    scores: np.ndarray
    group_code: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.y_true.shape[0])


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    tmp.replace(path)


def _group_code_for_spec(spec: Any) -> int:
    if spec.binary_label == "ATTACK":
        code = ATTACK_GROUP_CODES.get(spec.modeling_group_key)
        if code is None:
            raise FeatureExtractionError(
                f"unexpected validation attack group: {spec.modeling_group_key!r}"
            )
        return code
    bkey = benign_group_key(spec.benign_category, spec.modeling_group_key)
    code = BENIGN_GROUP_CODES.get(bkey or "")
    if code is None:
        raise FeatureExtractionError(
            f"unexpected validation benign group: "
            f"category={spec.benign_category!r} key={spec.modeling_group_key!r}"
        )
    return code


def require_phase2b2_complete(
    *,
    run_complete_path: Path | str | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = (project_root or PROJECT_ROOT).resolve()
    path = Path(run_complete_path or (DEFAULT_BASELINES_ROOT / "run_complete.json"))
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FeatureExtractionError(
            f"Phase 2B.2 run_complete.json missing: {path}. "
            "Run train-baselines first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise FeatureExtractionError(
            f"Phase 2B.2 not passed: status={payload.get('status')!r}"
        )
    if payload.get("smoke_only") is True:
        raise FeatureExtractionError(
            "refusing threshold sweep against a smoke-only 2B.2 run"
        )
    if int((payload.get("validation") or {}).get("rows_scored", -1)) != EXPECTED_VAL_ROWS:
        raise FeatureExtractionError(
            "Phase 2B.2 validation rows_scored != "
            f"{EXPECTED_VAL_ROWS}: {payload.get('validation')}"
        )
    return payload


def score_validation_tape(
    estimator: Any,
    *,
    project_root: Path,
    split_manifest_path: Path,
    expected_rows: int = EXPECTED_VAL_ROWS,
    feature_names: tuple[str, ...] | list[str] | None = None,
    progress_file: TextIO | None = None,
) -> ValidationScoreTape:
    """Score the full unsampled TRAIN-validation set once (no thresholding)."""
    specs = load_validation_specs(split_manifest_path, project_root=project_root)
    validate_validation_inventory(specs, smoke_only=False)

    y_true = np.empty(expected_rows, dtype=np.uint8)
    scores = np.empty(expected_rows, dtype=np.float32)
    group_code = np.empty(expected_rows, dtype=np.uint8)
    cursor = 0

    for batch in iter_validation_batches(
        specs,
        project_root=project_root,
        feature_names=feature_names,
    ):
        n = batch.X.shape[0]
        if cursor + n > y_true.shape[0]:
            need = cursor + n - y_true.shape[0]
            y_true = np.concatenate([y_true, np.empty(need, dtype=np.uint8)])
            scores = np.concatenate([scores, np.empty(need, dtype=np.float32)])
            group_code = np.concatenate([group_code, np.empty(need, dtype=np.uint8)])
        batch_scores = attack_score_from_estimator(estimator, batch.X)
        code = _group_code_for_spec(batch.spec)
        y_true[cursor : cursor + n] = batch.y
        scores[cursor : cursor + n] = batch_scores
        group_code[cursor : cursor + n] = code
        cursor += n
        if progress_file is not None and cursor % 500_000 < n:
            progress_file.write(f"  scored {cursor} validation rows\n")
            progress_file.flush()

    if cursor != expected_rows:
        raise FeatureExtractionError(
            f"validation score tape length {cursor} != expected {expected_rows}"
        )
    return ValidationScoreTape(
        y_true=y_true[:cursor],
        scores=scores[:cursor],
        group_code=group_code[:cursor],
    )


def _mask_metrics(
    scores: np.ndarray,
    y_true: np.ndarray,
    mask: np.ndarray,
    *,
    threshold: float,
    positive: bool,
) -> tuple[int, int, float | None]:
    """Return (hits, support, rate) for a subgroup mask.

    For attack groups (positive=True): hits=TP, rate=recall.
    For benign groups (positive=False): hits=FP, rate=FPR.
    """
    if not np.any(mask):
        return 0, 0, None
    sub_scores = scores[mask]
    sub_y = y_true[mask]
    support = int(sub_y.shape[0])
    pred_pos = sub_scores >= threshold
    if positive:
        # All should be attack; recall = mean(pred)
        tp = int(np.sum(pred_pos & (sub_y == 1)))
        pos = int(np.sum(sub_y == 1))
        return tp, pos, (float(tp) / pos if pos else None)
    fp = int(np.sum(pred_pos & (sub_y == 0)))
    neg = int(np.sum(sub_y == 0))
    return fp, neg, (float(fp) / neg if neg else None)


def metrics_at_threshold(
    tape: ValidationScoreTape,
    *,
    threshold: float,
    model_id: str,
    point_type: str,
    fpr_target: float | None = None,
) -> dict[str, Any]:
    y = tape.y_true
    s = tape.scores
    g = tape.group_code
    pred = s >= threshold

    benign = y == 0
    attack = y == 1
    fp = int(np.sum(pred & benign))
    tn = int(np.sum((~pred) & benign))
    benign_support = fp + tn
    benign_fpr = (fp / benign_support) if benign_support else None

    row: dict[str, Any] = {
        "model_id": model_id,
        "point_type": point_type,
        "threshold": float(threshold),
        "fpr_target": "" if fpr_target is None else float(fpr_target),
        "benign_fp": fp,
        "benign_support": benign_support,
        "benign_fpr": benign_fpr,
    }

    benign_cols = {
        "owltron_interaction": "owltron_interaction",
        "profiling_idle": "profiling_idle",
        "owltron_power": "owltron_power",
    }
    for gname, prefix in benign_cols.items():
        code = BENIGN_GROUP_CODES[gname]
        hits, support, rate = _mask_metrics(
            s, y, g == code, threshold=threshold, positive=False
        )
        row[f"{prefix}_fp"] = hits
        row[f"{prefix}_support"] = support
        row[f"{prefix}_fpr"] = rate

    attack_cols = {
        "DDoS|DDoS_TCP": "ddos_tcp",
        "DoS|DoS_TCP": "dos_tcp",
        "MQTT|MQTT_DoS_Publish_Flood": "mqtt_publish",
        "Recon|OS_Scan": "recon_os_scan",
    }
    recalls: list[float] = []
    for gname, prefix in attack_cols.items():
        code = ATTACK_GROUP_CODES[gname]
        tp, support, recall = _mask_metrics(
            s, y, g == code, threshold=threshold, positive=True
        )
        row[f"{prefix}_tp"] = tp
        row[f"{prefix}_support"] = support
        row[f"{prefix}_recall"] = recall
        if recall is not None:
            recalls.append(float(recall))

    row["macro_attack_group_recall"] = (
        float(sum(recalls) / len(recalls)) if recalls else None
    )
    row["min_attack_group_recall"] = float(min(recalls)) if recalls else None
    # Silence unused for lint clarity
    _ = attack
    return row


def threshold_for_benign_fpr(
    tape: ValidationScoreTape,
    fpr_target: float,
) -> float:
    """Smallest threshold whose empirical benign FPR is <= target.

    Uses the (1 - target) quantile of benign attack-scores as the operating
    point, then nudges upward if needed so exact FPR is <= target.
    """
    thr, _reached = threshold_for_benign_fpr_with_reachability(tape, fpr_target)
    return thr


def threshold_for_benign_fpr_with_reachability(
    tape: ValidationScoreTape,
    fpr_target: float,
) -> tuple[float, bool]:
    """Return (threshold, target_reached).

    If discrete scores prevent FPR <= target even at threshold=1.0,
    returns (1.0, False) — callers must not pretend the target was met.
    """
    if not (0.0 < fpr_target < 1.0):
        raise FeatureExtractionError(f"invalid fpr_target: {fpr_target}")
    benign_scores = tape.scores[tape.y_true == 0]
    if benign_scores.size == 0:
        raise FeatureExtractionError("no benign scores for FPR targeting")
    # Quantile: fraction of benign scores >= t should be ≈ fpr_target
    t = float(np.quantile(benign_scores.astype(np.float64), 1.0 - fpr_target))
    # Ensure FPR <= target (ties / discrete scores).
    fpr = float(np.mean(benign_scores >= t))
    if fpr <= fpr_target + 1e-15:
        return t, True
    # Raise to next distinct score above t among benign.
    above = benign_scores[benign_scores > t]
    if above.size == 0:
        # No higher threshold exists; report best effort at 1.0.
        fpr_at_one = float(np.mean(benign_scores >= 1.0))
        return 1.0, bool(fpr_at_one <= fpr_target + 1e-15)
    thr = float(np.min(above))
    fpr2 = float(np.mean(benign_scores >= thr))
    return thr, bool(fpr2 <= fpr_target + 1e-15)


def sweep_model(
    tape: ValidationScoreTape,
    *,
    model_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fixed_rows = [
        metrics_at_threshold(
            tape,
            threshold=t,
            model_id=model_id,
            point_type="fixed_threshold",
        )
        for t in FIXED_THRESHOLDS
    ]
    fpr_rows: list[dict[str, Any]] = []
    for target in BENIGN_FPR_TARGETS:
        thr = threshold_for_benign_fpr(tape, target)
        fpr_rows.append(
            metrics_at_threshold(
                tape,
                threshold=thr,
                model_id=model_id,
                point_type="fpr_target",
                fpr_target=target,
            )
        )
    return fixed_rows, fpr_rows


def run_threshold_sweep(
    *,
    project_root: Path | None = None,
    output_dir: Path | str | None = None,
    baselines_dir: Path | str | None = None,
    split_manifest_path: Path | str | None = None,
    progress_file: TextIO | None = None,
    cache_scores: bool = True,
) -> dict[str, Any]:
    """Rescore TRAIN-validation with frozen 2B.2 models; sweep thresholds."""
    root = (project_root or PROJECT_ROOT).resolve()
    base_dir = Path(baselines_dir or DEFAULT_BASELINES_ROOT)
    if not base_dir.is_absolute():
        base_dir = root / base_dir
    out = Path(output_dir or DEFAULT_SWEEP_ROOT)
    if not out.is_absolute():
        out = root / out

    b2_complete = require_phase2b2_complete(
        run_complete_path=base_dir / "run_complete.json",
        project_root=root,
    )
    contract = load_frozen_baseline_contract(
        base_dir / "baseline_contract.json",
        project_root=root,
    )

    split_path = Path(split_manifest_path or DEFAULT_SPLIT_MANIFEST_PATH)
    if not split_path.is_absolute():
        split_path = root / split_path

    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep_complete.json").unlink(missing_ok=True)

    contract_payload = {
        "strategy_version": THRESHOLD_SWEEP_VERSION,
        "parent_baseline_strategy": BASELINE_STRATEGY_VERSION,
        "task": "threshold_sweep_only",
        "retraining": False,
        "new_sampling": False,
        "test_access": False,
        "validation_sampling": "never",
        "validation_rows": EXPECTED_VAL_ROWS,
        "fixed_thresholds": list(FIXED_THRESHOLDS),
        "benign_fpr_targets": list(BENIGN_FPR_TARGETS),
        "label_mapping": dict(LABEL_MAPPING),
        "parent_run_complete": to_repo_relative(
            base_dir / "run_complete.json", project_root=root
        ),
        "baseline_contract_pins": contract.get("pins"),
        "note": (
            "Scores are predict_proba(ATTACK) from resampled FIT models; "
            "not calibrated real-world probabilities. Sweep only — do not "
            "freeze an operating point without review."
        ),
    }
    _atomic_json(out / "threshold_sweep_contract.json", contract_payload)

    all_fixed: list[dict[str, Any]] = []
    all_fpr: list[dict[str, Any]] = []
    model_summaries: list[dict[str, Any]] = []
    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for spec in MODEL_SPECS:
        model_id = str(spec["model_id"])
        model_path = base_dir / model_id / "models" / f"{model_id}.joblib"
        if not model_path.is_file():
            raise FeatureExtractionError(f"model artifact missing: {model_path}")
        if progress_file is not None:
            progress_file.write(f"Loading {model_id}...\n")
            progress_file.flush()
        estimator = joblib.load(model_path)

        score_cache = pred_dir / f"{model_id}_val_scores.npz"
        t0 = time.perf_counter()
        if cache_scores and score_cache.is_file():
            if progress_file is not None:
                progress_file.write(f"  reusing cached scores {score_cache}\n")
                progress_file.flush()
            cached = np.load(score_cache)
            tape = ValidationScoreTape(
                y_true=cached["y_true"],
                scores=cached["scores"],
                group_code=cached["group_code"],
            )
            if tape.n_rows != EXPECTED_VAL_ROWS:
                raise FeatureExtractionError(
                    f"cached score tape rows {tape.n_rows} != {EXPECTED_VAL_ROWS}"
                )
            score_seconds = 0.0
        else:
            if progress_file is not None:
                progress_file.write(
                    f"Scoring full TRAIN-validation for {model_id}...\n"
                )
                progress_file.flush()
            tape = score_validation_tape(
                estimator,
                project_root=root,
                split_manifest_path=split_path,
                progress_file=progress_file,
            )
            score_seconds = time.perf_counter() - t0
            if cache_scores:
                np.savez_compressed(
                    score_cache,
                    y_true=tape.y_true,
                    scores=tape.scores,
                    group_code=tape.group_code,
                )

        t1 = time.perf_counter()
        fixed_rows, fpr_rows = sweep_model(tape, model_id=model_id)
        sweep_seconds = time.perf_counter() - t1

        model_dir = out / model_id
        _atomic_csv(model_dir / "fixed_thresholds.csv", fixed_rows, list(SWEEP_ROW_COLUMNS))
        _atomic_csv(
            model_dir / "fpr_target_thresholds.csv", fpr_rows, list(SWEEP_ROW_COLUMNS)
        )
        all_fixed.extend(fixed_rows)
        all_fpr.extend(fpr_rows)
        model_summaries.append(
            {
                "model_id": model_id,
                "model_artifact": to_repo_relative(model_path, project_root=root),
                "model_artifact_sha256": file_sha256(model_path),
                "score_seconds": score_seconds,
                "sweep_seconds": sweep_seconds,
                "validation_rows": tape.n_rows,
                "score_cache": to_repo_relative(score_cache, project_root=root)
                if score_cache.is_file()
                else None,
            }
        )

    _atomic_csv(out / "comparison_fixed_thresholds.csv", all_fixed, list(SWEEP_ROW_COLUMNS))
    _atomic_csv(out / "comparison_fpr_targets.csv", all_fpr, list(SWEEP_ROW_COLUMNS))

    complete = {
        "status": "passed",
        "strategy_version": THRESHOLD_SWEEP_VERSION,
        "parent_baseline_strategy": BASELINE_STRATEGY_VERSION,
        "retraining": False,
        "new_sampling": False,
        "test_access": False,
        "validation_sampling": "never",
        "validation_rows": EXPECTED_VAL_ROWS,
        "fixed_thresholds": list(FIXED_THRESHOLDS),
        "benign_fpr_targets": list(BENIGN_FPR_TARGETS),
        "models": model_summaries,
        "parent_decision_threshold": b2_complete.get("decision_threshold"),
        "artifacts": {
            "threshold_sweep_contract": to_repo_relative(
                out / "threshold_sweep_contract.json", project_root=root
            ),
            "comparison_fixed_thresholds": to_repo_relative(
                out / "comparison_fixed_thresholds.csv", project_root=root
            ),
            "comparison_fpr_targets": to_repo_relative(
                out / "comparison_fpr_targets.csv", project_root=root
            ),
            "output_dir": to_repo_relative(out, project_root=root),
        },
        "next": (
            "Review operating-point tables (esp. ~0.94 vs Recon recall and "
            "Owltron interaction FPR). Do not freeze a threshold, retrain, "
            "or consult TEST without an explicit next-phase decision."
        ),
    }
    _atomic_json(out / "sweep_complete.json", complete)
    return complete


def format_threshold_sweep_summary(payload: dict[str, Any]) -> str:
    lines = [
        "Phase 2B.3A — threshold sweep (no retraining)",
        f"status: {payload.get('status')}",
        f"strategy_version: {payload.get('strategy_version')}",
        f"validation_rows: {payload.get('validation_rows')}",
        f"fixed_thresholds: {payload.get('fixed_thresholds')}",
        f"benign_fpr_targets: {payload.get('benign_fpr_targets')}",
    ]
    for m in payload.get("models") or []:
        lines.append(
            f"{m['model_id']}: score_s={m.get('score_seconds'):.1f} "
            f"sweep_s={m.get('sweep_seconds'):.3f}"
        )
    arts = payload.get("artifacts") or {}
    if arts.get("comparison_fixed_thresholds"):
        lines.append(f"fixed: {arts['comparison_fixed_thresholds']}")
    if arts.get("comparison_fpr_targets"):
        lines.append(f"fpr_targets: {arts['comparison_fpr_targets']}")
    lines.append(f"next: {payload.get('next')}")
    return "\n".join(lines) + "\n"
