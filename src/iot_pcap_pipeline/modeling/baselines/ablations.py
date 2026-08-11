"""Phase 2B.3B: HGB feature / class-weight ablations (no TEST)."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, TextIO

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight

from iot_pcap_pipeline.features.schema import V1_FEATURE_NAMES
from iot_pcap_pipeline.modeling.baselines.constants import (
    EXPECTED_FIT_ATTACK,
    EXPECTED_FIT_BENIGN,
    EXPECTED_FIT_ROWS,
    EXPECTED_VAL_ROWS,
    LABEL_MAPPING,
)
from iot_pcap_pipeline.modeling.baselines.contract import (
    DEFAULT_BASELINES_ROOT,
    load_frozen_baseline_contract,
    require_fit_view_ready,
)
from iot_pcap_pipeline.modeling.baselines.data import load_fit_arrays
from iot_pcap_pipeline.modeling.baselines.models import HGB_PARAMS, RANDOM_SEED
from iot_pcap_pipeline.modeling.baselines.threshold_sweep import (
    BENIGN_FPR_TARGETS,
    DEFAULT_SWEEP_ROOT,
    FIXED_THRESHOLDS,
    SWEEP_ROW_COLUMNS,
    ValidationScoreTape,
    require_phase2b2_complete,
    score_validation_tape,
    sweep_model,
)
from iot_pcap_pipeline.modeling.view import (
    DEFAULT_FIT_VIEW_MANIFEST_PATH,
    DEFAULT_SPLIT_MANIFEST_PATH,
    file_sha256,
)
from iot_pcap_pipeline.paths import (
    DEFAULT_MODELING_DIR,
    PROJECT_ROOT,
    to_repo_relative,
)
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError

ABLATION_VERSION = "phase2b3b_v1"
DEFAULT_ABLATION_ROOT = (
    DEFAULT_MODELING_DIR / "v1" / "baselines" / ABLATION_VERSION
)

# Drop only the five timing / span features for the 22-feature variants.
DROPPED_TEMPORAL_FEATURES: tuple[str, ...] = (
    "window_span_seconds",
    "iat_mean_seconds",
    "iat_std_seconds",
    "iat_p50_seconds",
    "iat_p95_seconds",
)
FEATURES_22: tuple[str, ...] = tuple(
    n for n in V1_FEATURE_NAMES if n not in DROPPED_TEMPORAL_FEATURES
)
assert len(FEATURES_22) == 22

# Primary comparison band from the 2B.3A review.
LOW_FPR_TARGETS: tuple[float, ...] = (0.01, 0.005, 0.001)

REFERENCE_HGB_AT_095: dict[str, float] = {
    "threshold": 0.95,
    "benign_fpr": 0.000844294,
    "ddos_tcp_recall": 0.998639,
    "dos_tcp_recall": 0.997767,
    "mqtt_publish_recall": 0.986201,
    "recon_os_scan_recall": 0.861194,
}


def balanced_class_weight_map(
    y: np.ndarray,
) -> dict[int, float]:
    """sklearn balanced weights on the FIT label vector."""
    classes = np.array([LABEL_MAPPING["BENIGN"], LABEL_MAPPING["ATTACK"]], dtype=np.int64)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights, strict=True)}


def build_hgb(*, class_weight: str | None) -> HistGradientBoostingClassifier:
    params = dict(HGB_PARAMS)
    params["class_weight"] = class_weight
    return HistGradientBoostingClassifier(**params)


VARIANT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "variant_id": "A_27_unweighted",
        "display_name": "HGB 27 features, unweighted (2B.2 baseline)",
        "feature_names": list(V1_FEATURE_NAMES),
        "class_weight": None,
        "reuse_from_2b2": True,
    },
    {
        "variant_id": "B_27_balanced",
        "display_name": "HGB 27 features, balanced weights",
        "feature_names": list(V1_FEATURE_NAMES),
        "class_weight": "balanced",
        "reuse_from_2b2": False,
    },
    {
        "variant_id": "C_22_unweighted",
        "display_name": "HGB 22 features (no timing), unweighted",
        "feature_names": list(FEATURES_22),
        "class_weight": None,
        "reuse_from_2b2": False,
    },
    {
        "variant_id": "D_22_balanced",
        "display_name": "HGB 22 features (no timing), balanced weights",
        "feature_names": list(FEATURES_22),
        "class_weight": "balanced",
        "reuse_from_2b2": False,
    },
)


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


def _select_feature_matrix(
    X_full: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    idxs = [list(V1_FEATURE_NAMES).index(name) for name in feature_names]
    return X_full[:, idxs]


def _load_or_score_tape(
    *,
    estimator: Any,
    variant_id: str,
    feature_names: list[str],
    pred_dir: Path,
    project_root: Path,
    split_path: Path,
    cache_scores: bool,
    progress_file: TextIO | None,
    reuse_2b3a_cache: Path | None = None,
) -> tuple[ValidationScoreTape, float]:
    score_cache = pred_dir / f"{variant_id}_val_scores.npz"
    t0 = time.perf_counter()
    if cache_scores and score_cache.is_file():
        if progress_file is not None:
            progress_file.write(f"  reusing {score_cache}\n")
            progress_file.flush()
        cached = np.load(score_cache)
        tape = ValidationScoreTape(
            y_true=cached["y_true"],
            scores=cached["scores"],
            group_code=cached["group_code"],
        )
        if tape.n_rows != EXPECTED_VAL_ROWS:
            raise FeatureExtractionError(
                f"{variant_id}: cached rows {tape.n_rows} != {EXPECTED_VAL_ROWS}"
            )
        return tape, 0.0

    if reuse_2b3a_cache is not None and reuse_2b3a_cache.is_file():
        if progress_file is not None:
            progress_file.write(f"  reusing 2B.3A cache {reuse_2b3a_cache}\n")
            progress_file.flush()
        cached = np.load(reuse_2b3a_cache)
        tape = ValidationScoreTape(
            y_true=cached["y_true"],
            scores=cached["scores"],
            group_code=cached["group_code"],
        )
        if tape.n_rows != EXPECTED_VAL_ROWS:
            raise FeatureExtractionError(
                f"{variant_id}: 2B.3A cache rows {tape.n_rows} != {EXPECTED_VAL_ROWS}"
            )
        if cache_scores:
            np.savez_compressed(
                score_cache,
                y_true=tape.y_true,
                scores=tape.scores,
                group_code=tape.group_code,
            )
        return tape, 0.0

    if progress_file is not None:
        progress_file.write(f"  scoring TRAIN-validation for {variant_id}...\n")
        progress_file.flush()
    tape = score_validation_tape(
        estimator,
        project_root=project_root,
        split_manifest_path=split_path,
        feature_names=feature_names,
        progress_file=progress_file,
    )
    elapsed = time.perf_counter() - t0
    if cache_scores:
        np.savez_compressed(
            score_cache,
            y_true=tape.y_true,
            scores=tape.scores,
            group_code=tape.group_code,
        )
    return tape, elapsed


def run_hgb_ablations(
    *,
    project_root: Path | None = None,
    output_dir: Path | str | None = None,
    baselines_dir: Path | str | None = None,
    sweep_dir_2b3a: Path | str | None = None,
    fit_manifest_path: Path | str | None = None,
    split_manifest_path: Path | str | None = None,
    progress_file: TextIO | None = None,
    cache_scores: bool = True,
) -> dict[str, Any]:
    """Train HGB variants B/C/D (reuse A), sweep thresholds, compare low-FPR."""
    root = (project_root or PROJECT_ROOT).resolve()
    base_dir = Path(baselines_dir or DEFAULT_BASELINES_ROOT)
    if not base_dir.is_absolute():
        base_dir = root / base_dir
    out = Path(output_dir or DEFAULT_ABLATION_ROOT)
    if not out.is_absolute():
        out = root / out
    sweep_2b3a = Path(sweep_dir_2b3a or DEFAULT_SWEEP_ROOT)
    if not sweep_2b3a.is_absolute():
        sweep_2b3a = root / sweep_2b3a

    require_fit_view_ready(project_root=root, smoke_only=False)
    require_phase2b2_complete(
        run_complete_path=base_dir / "run_complete.json",
        project_root=root,
    )
    contract = load_frozen_baseline_contract(
        base_dir / "baseline_contract.json",
        project_root=root,
    )

    fit_man = Path(fit_manifest_path or DEFAULT_FIT_VIEW_MANIFEST_PATH)
    if not fit_man.is_absolute():
        fit_man = root / fit_man
    split_path = Path(split_manifest_path or DEFAULT_SPLIT_MANIFEST_PATH)
    if not split_path.is_absolute():
        split_path = root / split_path

    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation_complete.json").unlink(missing_ok=True)
    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if progress_file is not None:
        progress_file.write("Loading FIT arrays (27 features)...\n")
        progress_file.flush()
    fit = load_fit_arrays(fit_man, project_root=root, smoke_only=False)
    if fit.n_rows != EXPECTED_FIT_ROWS:
        raise FeatureExtractionError(f"FIT rows {fit.n_rows} != {EXPECTED_FIT_ROWS}")
    weight_map = balanced_class_weight_map(fit.y)
    expected_benign_w = EXPECTED_FIT_ROWS / (2.0 * EXPECTED_FIT_BENIGN)
    expected_attack_w = EXPECTED_FIT_ROWS / (2.0 * EXPECTED_FIT_ATTACK)

    contract_payload = {
        "strategy_version": ABLATION_VERSION,
        "parent_baseline_strategy": "phase2b2_v1",
        "parent_threshold_sweep": "phase2b3a_v1",
        "model_family": "HistGradientBoostingClassifier",
        "hgb_params": dict(HGB_PARAMS),
        "random_state": RANDOM_SEED,
        "dropped_temporal_features": list(DROPPED_TEMPORAL_FEATURES),
        "features_22": list(FEATURES_22),
        "balanced_class_weights_on_fit": {
            "BENIGN": weight_map[LABEL_MAPPING["BENIGN"]],
            "ATTACK": weight_map[LABEL_MAPPING["ATTACK"]],
            "formula": "n_samples / (n_classes * n_class)",
            "approx_expected": {
                "BENIGN": expected_benign_w,
                "ATTACK": expected_attack_w,
            },
        },
        "variants": [
            {
                "variant_id": v["variant_id"],
                "feature_count": len(v["feature_names"]),
                "class_weight": v["class_weight"],
                "reuse_from_2b2": v["reuse_from_2b2"],
            }
            for v in VARIANT_SPECS
        ],
        "low_fpr_targets": list(LOW_FPR_TARGETS),
        "fixed_thresholds": list(FIXED_THRESHOLDS),
        "test_access": False,
        "validation_sampling": "never",
        "baseline_contract_pins": contract.get("pins"),
        "reference_hgb_at_0_95": REFERENCE_HGB_AT_095,
        "improvement_rule": (
            "Replace A only if another variant materially improves Recon recall "
            "while staying around <=0.5–1% benign FPR without a substantial MQTT hit."
        ),
    }
    _atomic_json(out / "ablation_contract.json", contract_payload)

    all_fixed: list[dict[str, Any]] = []
    all_fpr: list[dict[str, Any]] = []
    low_fpr_rows: list[dict[str, Any]] = []
    variant_summaries: list[dict[str, Any]] = []

    baseline_hgb = base_dir / "hist_gradient_boosting" / "models" / "hist_gradient_boosting.joblib"
    cache_a_2b3a = sweep_2b3a / "predictions" / "hist_gradient_boosting_val_scores.npz"

    for spec in VARIANT_SPECS:
        variant_id = str(spec["variant_id"])
        feature_names = list(spec["feature_names"])
        class_weight = spec["class_weight"]
        if progress_file is not None:
            progress_file.write(f"\n=== {variant_id} ===\n")
            progress_file.flush()

        model_path = models_dir / f"{variant_id}.joblib"
        fit_seconds = 0.0
        if spec["reuse_from_2b2"]:
            if not baseline_hgb.is_file():
                raise FeatureExtractionError(f"baseline HGB missing: {baseline_hgb}")
            estimator = joblib.load(baseline_hgb)
            # Mirror into ablation models/ for a self-contained tree.
            joblib.dump(estimator, model_path)
            reused = True
        else:
            reused = False
            X = _select_feature_matrix(fit.X, feature_names)
            estimator = build_hgb(class_weight=class_weight)
            t0 = time.perf_counter()
            if progress_file is not None:
                progress_file.write(
                    f"  fitting HGB features={len(feature_names)} "
                    f"class_weight={class_weight!r}...\n"
                )
                progress_file.flush()
            estimator.fit(X, fit.y)
            fit_seconds = time.perf_counter() - t0
            joblib.dump(estimator, model_path)

        tape, score_seconds = _load_or_score_tape(
            estimator=estimator,
            variant_id=variant_id,
            feature_names=feature_names,
            pred_dir=pred_dir,
            project_root=root,
            split_path=split_path,
            cache_scores=cache_scores,
            progress_file=progress_file,
            reuse_2b3a_cache=cache_a_2b3a if spec["reuse_from_2b2"] else None,
        )
        t1 = time.perf_counter()
        fixed_rows, fpr_rows = sweep_model(tape, model_id=variant_id)
        sweep_seconds = time.perf_counter() - t1

        # Annotate with variant metadata for comparison CSVs.
        for row in fixed_rows + fpr_rows:
            row["variant_id"] = variant_id
            row["feature_count"] = len(feature_names)
            row["class_weight"] = (
                "none" if class_weight is None else str(class_weight)
            )

        variant_dir = out / variant_id
        _atomic_csv(
            variant_dir / "fixed_thresholds.csv", fixed_rows, list(SWEEP_ROW_COLUMNS)
        )
        _atomic_csv(
            variant_dir / "fpr_target_thresholds.csv",
            fpr_rows,
            list(SWEEP_ROW_COLUMNS),
        )
        meta = {
            "variant_id": variant_id,
            "display_name": spec["display_name"],
            "feature_names": feature_names,
            "feature_count": len(feature_names),
            "dropped_features": [
                f for f in DROPPED_TEMPORAL_FEATURES if f not in feature_names
            ],
            "class_weight": class_weight,
            "balanced_weights_applied": weight_map if class_weight == "balanced" else None,
            "hgb_params": dict(HGB_PARAMS),
            "reuse_from_2b2": bool(spec["reuse_from_2b2"]),
            "fit_seconds": fit_seconds,
            "score_seconds": score_seconds,
            "sweep_seconds": sweep_seconds,
            "model_artifact": to_repo_relative(model_path, project_root=root),
            "model_artifact_sha256": file_sha256(model_path),
            "fit_rows": fit.n_rows,
            "fit_attack_rows": fit.n_attack,
            "fit_benign_rows": fit.n_benign,
        }
        _atomic_json(variant_dir / "variant_metadata.json", meta)

        all_fixed.extend(fixed_rows)
        all_fpr.extend(fpr_rows)
        for target in LOW_FPR_TARGETS:
            match = next(
                (
                    r
                    for r in fpr_rows
                    if r.get("fpr_target") != ""
                    and abs(float(r["fpr_target"]) - target) < 1e-12
                ),
                None,
            )
            if match is None:
                raise FeatureExtractionError(
                    f"{variant_id}: missing FPR target row for {target}"
                )
            low_fpr_rows.append(dict(match))

        variant_summaries.append(
            {
                "variant_id": variant_id,
                "feature_count": len(feature_names),
                "class_weight": class_weight,
                "reused_2b2": reused,
                "fit_seconds": fit_seconds,
                "score_seconds": score_seconds,
                "sweep_seconds": sweep_seconds,
            }
        )

    compare_cols = ["variant_id", "feature_count", "class_weight", *SWEEP_ROW_COLUMNS]
    # Rebuild rows with variant fields first for readability.
    def _ordered(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out_rows = []
        for r in rows:
            out_rows.append({c: r.get(c, "") for c in compare_cols})
        return out_rows

    _atomic_csv(
        out / "comparison_fixed_thresholds.csv",
        _ordered(all_fixed),
        compare_cols,
    )
    _atomic_csv(
        out / "comparison_fpr_targets.csv",
        _ordered(all_fpr),
        compare_cols,
    )
    _atomic_csv(
        out / "comparison_low_fpr.csv",
        _ordered(low_fpr_rows),
        compare_cols,
    )

    # Decision helper: does any non-A variant beat A on Recon at <=1% / 0.5% FPR?
    decision = _summarize_decision(low_fpr_rows)

    complete = {
        "status": "passed",
        "strategy_version": ABLATION_VERSION,
        "variants": variant_summaries,
        "balanced_class_weights_on_fit": {
            "BENIGN": weight_map[LABEL_MAPPING["BENIGN"]],
            "ATTACK": weight_map[LABEL_MAPPING["ATTACK"]],
        },
        "low_fpr_targets": list(LOW_FPR_TARGETS),
        "decision_preview": decision,
        "test_access": False,
        "validation_sampling": "never",
        "artifacts": {
            "ablation_contract": to_repo_relative(
                out / "ablation_contract.json", project_root=root
            ),
            "comparison_low_fpr": to_repo_relative(
                out / "comparison_low_fpr.csv", project_root=root
            ),
            "comparison_fpr_targets": to_repo_relative(
                out / "comparison_fpr_targets.csv", project_root=root
            ),
            "comparison_fixed_thresholds": to_repo_relative(
                out / "comparison_fixed_thresholds.csv", project_root=root
            ),
            "output_dir": to_repo_relative(out, project_root=root),
        },
        "next": (
            "Review comparison_low_fpr.csv. Replace A only if Recon improves "
            "materially at <=0.5–1% benign FPR without a substantial MQTT hit; "
            "otherwise keep 27-feature unweighted HGB and provisionally use ~0.95. "
            "Do not touch TEST."
        ),
    }
    _atomic_json(out / "ablation_complete.json", complete)
    return complete


def _summarize_decision(low_fpr_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Non-binding preview against the improvement rule (human still decides)."""
    by_variant: dict[str, dict[float, dict[str, Any]]] = {}
    for row in low_fpr_rows:
        vid = str(row["variant_id"])
        target = float(row["fpr_target"])
        by_variant.setdefault(vid, {})[target] = row

    a = by_variant.get("A_27_unweighted", {})
    notes: list[str] = []
    challengers: list[dict[str, Any]] = []
    for vid, targets in by_variant.items():
        if vid == "A_27_unweighted":
            continue
        for fpr_t in (0.01, 0.005):
            if fpr_t not in targets or fpr_t not in a:
                continue
            cand = targets[fpr_t]
            base = a[fpr_t]
            recon_gain = float(cand["recon_os_scan_recall"]) - float(
                base["recon_os_scan_recall"]
            )
            mqtt_delta = float(cand["mqtt_publish_recall"]) - float(
                base["mqtt_publish_recall"]
            )
            entry = {
                "variant_id": vid,
                "fpr_target": fpr_t,
                "recon_gain": recon_gain,
                "mqtt_delta": mqtt_delta,
                "benign_fpr": cand["benign_fpr"],
                "recon_os_scan_recall": cand["recon_os_scan_recall"],
                "mqtt_publish_recall": cand["mqtt_publish_recall"],
            }
            # Heuristic only: +2pp Recon, MQTT drop < 1pp, FPR still <= target band.
            if recon_gain >= 0.02 and mqtt_delta >= -0.01:
                entry["meets_heuristic"] = True
                challengers.append(entry)
            else:
                entry["meets_heuristic"] = False
                challengers.append(entry)

    if any(c.get("meets_heuristic") for c in challengers):
        notes.append(
            "At least one variant meets the coarse Recon-up / MQTT-stable heuristic; "
            "human review required before replacing A."
        )
        recommendation = "review_challengers"
    else:
        notes.append(
            "No variant meets the coarse heuristic vs A at 1% or 0.5% FPR targets; "
            "default path is keep A (27 unweighted) and provisional ~0.95."
        )
        recommendation = "keep_A_provisional_0_95"

    return {
        "recommendation": recommendation,
        "challengers": challengers,
        "notes": notes,
    }


def format_ablation_summary(payload: dict[str, Any]) -> str:
    lines = [
        "Phase 2B.3B — HGB feature / weight ablations",
        f"status: {payload.get('status')}",
        f"strategy_version: {payload.get('strategy_version')}",
    ]
    weights = payload.get("balanced_class_weights_on_fit") or {}
    if weights:
        lines.append(
            f"balanced_weights: BENIGN={weights.get('BENIGN'):.4f} "
            f"ATTACK={weights.get('ATTACK'):.4f}"
        )
    for v in payload.get("variants") or []:
        lines.append(
            f"{v['variant_id']}: features={v['feature_count']} "
            f"weight={v['class_weight']!r} fit_s={v.get('fit_seconds'):.1f} "
            f"score_s={v.get('score_seconds'):.1f}"
        )
    decision = payload.get("decision_preview") or {}
    lines.append(f"decision_preview: {decision.get('recommendation')}")
    for note in decision.get("notes") or []:
        lines.append(f"  note: {note}")
    arts = payload.get("artifacts") or {}
    if arts.get("comparison_low_fpr"):
        lines.append(f"low_fpr: {arts['comparison_low_fpr']}")
    lines.append(f"next: {payload.get('next')}")
    return "\n".join(lines) + "\n"
