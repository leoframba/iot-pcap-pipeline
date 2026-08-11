"""Phase 2B.4C: external boosting challengers (XGBoost + CatBoost).

Fixed untuned configs only. No early stopping against TRAIN-validation.
Feature selection remains deferred (all 27). TEST stays sealed.
"""

from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path
from typing import Any, TextIO

import joblib
import numpy as np

from iot_pcap_pipeline.features.parquet import feature_schema_sha256
from iot_pcap_pipeline.features.schema import DEFAULT_FEATURE_SCHEMA_PATH, V1_FEATURE_NAMES
from iot_pcap_pipeline.modeling.baselines.constants import (
    EXPECTED_FIT_ATTACK,
    EXPECTED_FIT_BENIGN,
    EXPECTED_FIT_PCAPS,
    EXPECTED_FIT_ROWS,
    EXPECTED_VAL_PCAPS,
    EXPECTED_VAL_ROWS,
    LABEL_MAPPING,
)
from iot_pcap_pipeline.modeling.baselines.contract import require_fit_view_ready
from iot_pcap_pipeline.modeling.baselines.data import load_fit_arrays
from iot_pcap_pipeline.modeling.baselines.extratrees import (
    COMPARISON_COLUMNS,
    DEFAULT_EXTRA_TREES_ROOT,
    EXTRA_TREES_VERSION,
    _parent_low_fpr_rows,
    assert_fit_arrays_ready,
    require_phase2b4_complete,
)
from iot_pcap_pipeline.modeling.baselines.model_family import (
    BAKEOFF_SWEEP_COLUMNS,
    DEFAULT_MODEL_FAMILY_ROOT,
    MODEL_FAMILY_VERSION,
    LOW_FPR_TARGETS,
    RANKING_CRITERIA,
    _load_score_cache,
    _save_score_cache,
    _write_family_artifacts,
    metrics_at_threshold_bakeoff,
    score_bakeoff_tape,
)
from iot_pcap_pipeline.modeling.baselines.models import (
    CATBOOST_PARAMS,
    RANDOM_SEED,
    XGBOOST_PARAMS,
    attack_score_from_estimator,
    build_catboost,
    build_xgboost,
)
from iot_pcap_pipeline.modeling.baselines.threshold_sweep import (
    FIXED_THRESHOLDS,
    threshold_for_benign_fpr_with_reachability,
)
from iot_pcap_pipeline.modeling.freeze import FROZEN_SAMPLING_PLAN_ID
from iot_pcap_pipeline.modeling.view import (
    DEFAULT_FIT_VIEW_MANIFEST_PATH,
    DEFAULT_SPLIT_MANIFEST_PATH,
    DEFAULT_TRAINING_VIEW_CONTRACT_PATH,
    file_sha256,
)
from iot_pcap_pipeline.paths import (
    DEFAULT_MODELING_DIR,
    PROJECT_ROOT,
    to_repo_relative,
)
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError

EXTERNAL_BOOST_VERSION = "phase2b4c_v1"
DEFAULT_EXTERNAL_BOOST_ROOT = (
    DEFAULT_MODELING_DIR / "v1" / "baselines" / EXTERNAL_BOOST_VERSION
)

DECISION_RULE: list[str] = [
    "At matched low FPR, rank by: (1) min attack-family recall, (2) Recon, "
    "(3) MQTT, (4) macro attack recall, (5) Owltron FPR.",
    "Practical band: ≤0.5% and ≤0.1% benign FPR.",
    "No hyperparameter search; fixed configs only.",
    "No early stopping / best-iteration selection on TRAIN-validation.",
    "Do not auto-replace HGB; stop for review.",
]

CANDIDATE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "model_id": "xgboost",
        "display_name": "XGBClassifier (fixed hist config)",
        "builder": build_xgboost,
        "hyperparameters": dict(XGBOOST_PARAMS),
        "notes": [
            "No early_stopping_rounds.",
            "No eval_set against TRAIN-validation.",
        ],
    },
    {
        "model_id": "catboost",
        "display_name": "CatBoostClassifier (fixed Logloss config)",
        "builder": build_catboost,
        "hyperparameters": dict(CATBOOST_PARAMS),
        "notes": [
            "No use_best_model / eval_set against TRAIN-validation.",
            "allow_writing_files=False.",
        ],
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
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    tmp.replace(path)


def _package_versions() -> dict[str, str]:
    import catboost
    import numpy
    import pyarrow
    import sklearn
    import xgboost

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
        "pyarrow": pyarrow.__version__,
        "xgboost": xgboost.__version__,
        "catboost": catboost.__version__,
        "platform": platform.platform(),
    }


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_phase2b4b_optional(
    *,
    project_root: Path | None = None,
) -> dict[str, Any] | None:
    """Prefer 2B.4B four-family table if present; otherwise None."""
    root = (project_root or PROJECT_ROOT).resolve()
    path = root / DEFAULT_EXTRA_TREES_ROOT / "extratrees_complete.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        return None
    if payload.get("strategy_version") != EXTRA_TREES_VERSION:
        return None
    return payload


def build_external_boost_contract(
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = (project_root or PROJECT_ROOT).resolve()
    require_fit_view_ready(project_root=root, smoke_only=False)
    parent = require_phase2b4_complete(project_root=root)
    parent_et = require_phase2b4b_optional(project_root=root)
    parent_contract_path = root / DEFAULT_MODEL_FAMILY_ROOT / "model_family_contract.json"
    parent_contract = json.loads(parent_contract_path.read_text(encoding="utf-8"))

    fit_man = root / DEFAULT_FIT_VIEW_MANIFEST_PATH
    train_contract = root / DEFAULT_TRAINING_VIEW_CONTRACT_PATH
    split_path = root / DEFAULT_SPLIT_MANIFEST_PATH
    schema_path = root / DEFAULT_FEATURE_SCHEMA_PATH

    pins = dict(parent_contract.get("pins") or {})
    live_pins = {
        "feature_schema_sha256": feature_schema_sha256(schema_path),
        "training_view_contract_sha256": file_sha256(train_contract),
        "fit_view_manifest_sha256": file_sha256(fit_man),
        "modeling_split_manifest_sha256": file_sha256(split_path),
    }
    for key, value in live_pins.items():
        if pins.get(key) != value:
            raise FeatureExtractionError(
                f"pin drift vs Phase 2B.4 for {key}: "
                f"parent={pins.get(key)!r} live={value!r}"
            )

    parent_low = (
        root / DEFAULT_EXTRA_TREES_ROOT / "comparison_low_fpr.csv"
        if parent_et is not None
        else root / DEFAULT_MODEL_FAMILY_ROOT / "comparison_low_fpr.csv"
    )
    if not parent_low.is_file():
        raise FeatureExtractionError(f"missing parent low-FPR table: {parent_low}")

    return {
        "strategy_version": EXTERNAL_BOOST_VERSION,
        "parent_strategy": MODEL_FAMILY_VERSION,
        "parent_extratrees_strategy": EXTRA_TREES_VERSION if parent_et else None,
        "purpose": "external_boosting_challengers",
        "status": "frozen",
        "feature_selection_status": "deferred",
        "bakeoff_feature_set": "all_27_v1",
        "final_feature_count": "unresolved",
        "feature_count": 27,
        "feature_names": list(V1_FEATURE_NAMES),
        "label_mapping": dict(LABEL_MAPPING),
        "class_weights": "none",
        "threshold_selection": False,
        "hyperparameter_search": False,
        "early_stopping_on_validation": False,
        "test_access": False,
        "training_view": FROZEN_SAMPLING_PLAN_ID,
        "fit": {
            "pcaps": EXPECTED_FIT_PCAPS,
            "rows": EXPECTED_FIT_ROWS,
            "attack_rows": EXPECTED_FIT_ATTACK,
            "benign_rows": EXPECTED_FIT_BENIGN,
            "sampling": "group_balanced",
        },
        "validation": {
            "pcaps": EXPECTED_VAL_PCAPS,
            "rows": EXPECTED_VAL_ROWS,
            "sampling": "never",
        },
        "test": {"access": False, "pcaps_read": 0},
        "candidates": {
            "xgboost": {
                "hyperparameters": dict(XGBOOST_PARAMS),
                "early_stopping": False,
            },
            "catboost": {
                "hyperparameters": dict(CATBOOST_PARAMS),
                "early_stopping": False,
                "use_best_model": False,
            },
        },
        "low_fpr_targets": list(LOW_FPR_TARGETS),
        "fixed_thresholds": list(FIXED_THRESHOLDS),
        "ranking_criteria": list(RANKING_CRITERIA),
        "decision_rule": list(DECISION_RULE),
        "auto_declare_winner": False,
        "pins": {
            **live_pins,
            "parent_model_family_complete_status": parent.get("status"),
            "parent_hgb_model_artifact_sha256": pins.get("hgb_model_artifact_sha256"),
        },
        "artifacts": {
            "feature_schema": to_repo_relative(schema_path, project_root=root),
            "training_view_contract": to_repo_relative(
                train_contract, project_root=root
            ),
            "fit_view_manifest": to_repo_relative(fit_man, project_root=root),
            "modeling_split_manifest": to_repo_relative(split_path, project_root=root),
            "parent_model_family_contract": to_repo_relative(
                parent_contract_path, project_root=root
            ),
            "parent_comparison_low_fpr": to_repo_relative(parent_low, project_root=root),
        },
        "scope_limits": [
            "Exactly XGBoost and CatBoost; fixed configs only.",
            "No grids, no early stopping on TRAIN-validation.",
            "Feature 22-vs-27 remains deferred.",
            "No TEST access.",
            "Do not auto-replace HGB.",
        ],
    }


def prepare_external_boost_challengers(
    *,
    project_root: Path | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    root = (project_root or PROJECT_ROOT).resolve()
    out = Path(output_dir or DEFAULT_EXTERNAL_BOOST_ROOT)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    (out / "external_boost_complete.json").unlink(missing_ok=True)
    payload = build_external_boost_contract(project_root=root)
    contract_path = out / "external_boost_contract.json"
    _atomic_json(contract_path, payload)
    return {
        "status": "prepared",
        "strategy_version": EXTERNAL_BOOST_VERSION,
        "contract_path": to_repo_relative(contract_path, project_root=root),
        "feature_selection_status": payload["feature_selection_status"],
        "next": (
            "Review external_boost_contract.json, then run "
            "`iot-pcap-pipeline run-external-boost-challengers`."
        ),
    }


def load_external_boost_contract(
    path: Path | str,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = (project_root or PROJECT_ROOT).resolve()
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    if not p.is_file():
        raise FeatureExtractionError(
            f"external_boost_contract.json missing: {p}. "
            "Run prepare-external-boost-challengers first."
        )
    payload = json.loads(p.read_text(encoding="utf-8"))
    if payload.get("strategy_version") != EXTERNAL_BOOST_VERSION:
        raise FeatureExtractionError(
            f"unexpected strategy_version: {payload.get('strategy_version')!r}"
        )
    if payload.get("feature_selection_status") != "deferred":
        raise FeatureExtractionError("feature_selection_status must be deferred")
    if int(payload.get("feature_count") or 0) != 27:
        raise FeatureExtractionError("feature_count must be 27")
    if payload.get("early_stopping_on_validation") is not False:
        raise FeatureExtractionError("early_stopping_on_validation must be false")
    if payload.get("test_access") is not False:
        raise FeatureExtractionError("test_access must be false")
    return payload


def _assert_complete_gate(payload: dict[str, Any]) -> None:
    issues: list[str] = []
    fit = payload.get("fit") or {}
    val = payload.get("validation") or {}
    test = payload.get("test") or {}
    if int(fit.get("rows") or -1) != EXPECTED_FIT_ROWS:
        issues.append("wrong FIT totals")
    if int(fit.get("pcaps") or -1) != EXPECTED_FIT_PCAPS:
        issues.append("wrong FIT PCAPs")
    if int(val.get("rows_scored_per_model") or -1) != EXPECTED_VAL_ROWS:
        issues.append("wrong validation totals")
    if int(val.get("pcaps_scored_per_model") or -1) != EXPECTED_VAL_PCAPS:
        issues.append("wrong validation PCAPs")
    if val.get("sampling") != "never":
        issues.append("validation must be unsampled")
    if int(payload.get("feature_count") or -1) != 27:
        issues.append("feature_count != 27")
    if test.get("access") is not False or int(test.get("pcaps_read", -1)) != 0:
        issues.append("TEST access")
    if payload.get("early_stopping_on_validation") is not False:
        issues.append("early stopping enabled")
    models = payload.get("models") or []
    ids = [m.get("model_id") for m in models]
    if ids != ["xgboost", "catboost"]:
        issues.append("models must be xgboost and catboost")
    for m in models:
        sha = str(m.get("model_artifact_sha256") or "")
        if len(sha) != 64:
            issues.append(f"missing hash for {m.get('model_id')}")
    arts = payload.get("artifacts") or {}
    for key in (
        "comparison_low_fpr",
        "comparison_fixed_thresholds",
        "comparison_ranking_metrics",
    ):
        if not arts.get(key):
            issues.append(f"missing artifact {key}")
    if issues:
        raise FeatureExtractionError(
            "external_boost_complete acceptance gate failed: " + "; ".join(issues)
        )


def run_external_boost_challengers(
    *,
    project_root: Path | None = None,
    output_dir: Path | str | None = None,
    fit_manifest_path: Path | str | None = None,
    split_manifest_path: Path | str | None = None,
    progress_file: TextIO | None = None,
    cache_scores: bool = True,
) -> dict[str, Any]:
    """Fit XGBoost + CatBoost once each; score full TRAIN-validation; compare."""
    root = (project_root or PROJECT_ROOT).resolve()
    out = Path(output_dir or DEFAULT_EXTERNAL_BOOST_ROOT)
    if not out.is_absolute():
        out = root / out

    contract = load_external_boost_contract(
        out / "external_boost_contract.json", project_root=root
    )
    require_fit_view_ready(project_root=root, smoke_only=False)
    require_phase2b4_complete(project_root=root)

    fit_man = Path(fit_manifest_path or DEFAULT_FIT_VIEW_MANIFEST_PATH)
    if not fit_man.is_absolute():
        fit_man = root / fit_man
    split_path = Path(split_manifest_path or DEFAULT_SPLIT_MANIFEST_PATH)
    if not split_path.is_absolute():
        split_path = root / split_path

    out.mkdir(parents=True, exist_ok=True)
    (out / "external_boost_complete.json").unlink(missing_ok=True)
    models_dir = out / "models"
    pred_dir = out / "predictions"
    models_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    if progress_file is not None:
        progress_file.write("Loading FIT arrays (27 features)...\n")
        progress_file.flush()
    fit = load_fit_arrays(fit_man, project_root=root, smoke_only=False)
    assert_fit_arrays_ready(fit)

    parent_low_rel = (contract.get("artifacts") or {}).get("parent_comparison_low_fpr")
    parent_low = root / str(parent_low_rel)
    if not parent_low.is_file():
        raise FeatureExtractionError(f"missing parent low-FPR table: {parent_low}")

    all_fixed: list[dict[str, Any]] = []
    new_low_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    model_summaries: list[dict[str, Any]] = []
    reachability: list[dict[str, Any]] = []

    # Seed ranking with prior secondary metrics when available.
    for prior in (
        root / DEFAULT_MODEL_FAMILY_ROOT / "comparison_ranking_metrics.csv",
        root / DEFAULT_EXTRA_TREES_ROOT / "comparison_ranking_metrics.csv",
    ):
        if prior.is_file():
            for row in _read_csv_rows(prior):
                # Prefer later files' rows for duplicate model_ids.
                ranking_rows = [
                    r for r in ranking_rows if r.get("model_id") != row.get("model_id")
                ]
                ranking_rows.append(row)

    for spec in CANDIDATE_SPECS:
        model_id = str(spec["model_id"])
        model_dir = out / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{model_id}.joblib"
        score_cache = pred_dir / f"{model_id}_val_scores.npz"

        if progress_file is not None:
            progress_file.write(f"\n=== {model_id} ===\n")
            progress_file.write(f"  fitting {model_id} (no early stopping)...\n")
            progress_file.flush()

        builder = spec["builder"]
        estimator = builder()
        t0 = time.perf_counter()
        # Explicit: fit on FIT only — never pass eval_set / validation.
        estimator.fit(fit.X, fit.y)
        fit_seconds = time.perf_counter() - t0
        joblib.dump(estimator, model_path)
        model_sha = file_sha256(model_path)
        _ = attack_score_from_estimator(estimator, fit.X[: min(16, fit.n_rows)])

        if cache_scores and score_cache.is_file():
            if progress_file is not None:
                progress_file.write(f"  loading score cache {score_cache.name}\n")
                progress_file.flush()
            tape = _load_score_cache(score_cache)
            score_seconds = 0.0
        else:
            if progress_file is not None:
                progress_file.write(
                    f"  scoring TRAIN-validation for {model_id}...\n"
                )
                progress_file.flush()
            t1 = time.perf_counter()
            tape = score_bakeoff_tape(
                estimator,
                project_root=root,
                split_manifest_path=split_path,
                progress_file=progress_file,
            )
            score_seconds = time.perf_counter() - t1
            if cache_scores:
                _save_score_cache(score_cache, tape)

        if tape.n_rows != EXPECTED_VAL_ROWS:
            raise FeatureExtractionError(
                f"{model_id}: rows_scored={tape.n_rows} != {EXPECTED_VAL_ROWS}"
            )
        if len(tape.pcap_table) != EXPECTED_VAL_PCAPS:
            raise FeatureExtractionError(
                f"{model_id}: pcaps_scored={len(tape.pcap_table)} != "
                f"{EXPECTED_VAL_PCAPS}"
            )

        art = _write_family_artifacts(
            model_dir=model_dir,
            model_id=model_id,
            display_name=str(spec["display_name"]),
            hyperparameters=dict(spec["hyperparameters"]),
            model_path=model_path,
            model_sha=model_sha,
            fit_seconds=fit_seconds,
            score_seconds=score_seconds,
            tape=tape,
            project_root=root,
            extra_metadata={
                "parent_strategy": MODEL_FAMILY_VERSION,
                "purpose": "external_boosting_challengers",
                "early_stopping_on_validation": False,
                "selection_role_at_0_5": "secondary",
                "notes": list(spec.get("notes") or []),
                "versions": _package_versions(),
            },
            reused=False,
            strategy_version=EXTERNAL_BOOST_VERSION,
        )
        metrics_path = model_dir / "metrics.json"
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics_payload["selection_role"] = "secondary"
        metrics_payload["early_stopping_on_validation"] = False
        metrics_payload["note"] = (
            "Threshold-0.5 / ROC-AUC / PR-AUC are secondary; selection uses "
            "comparison_low_fpr.csv. No early stopping was used."
        )
        _atomic_json(metrics_path, metrics_payload)

        fixed_rows = [
            metrics_at_threshold_bakeoff(
                tape,
                threshold=t,
                model_id=model_id,
                point_type="fixed_threshold",
            )
            for t in FIXED_THRESHOLDS
        ]
        for row in fixed_rows:
            row["target_reached"] = ""
        all_fixed.extend(fixed_rows)

        for target in LOW_FPR_TARGETS:
            thr, reached = threshold_for_benign_fpr_with_reachability(
                tape.as_validation_tape(), target
            )
            row = metrics_at_threshold_bakeoff(
                tape,
                threshold=thr,
                model_id=model_id,
                point_type="fpr_target",
                fpr_target=target,
            )
            row["target_reached"] = bool(reached)
            new_low_rows.append(
                {
                    "model_id": model_id,
                    "fpr_target": row["fpr_target"],
                    "target_reached": row["target_reached"],
                    "threshold": row["threshold"],
                    "benign_fp": row["benign_fp"],
                    "benign_support": row["benign_support"],
                    "benign_fpr": row["benign_fpr"],
                    "ddos_tcp_recall": row["ddos_tcp_recall"],
                    "dos_tcp_recall": row["dos_tcp_recall"],
                    "mqtt_publish_recall": row["mqtt_publish_recall"],
                    "recon_os_scan_recall": row["recon_os_scan_recall"],
                    "macro_attack_group_recall": row["macro_attack_group_recall"],
                    "min_attack_group_recall": row["min_attack_group_recall"],
                    "owltron_interaction_fpr": row["owltron_interaction_fpr"],
                    "profiling_idle_fpr": row["profiling_idle_fpr"],
                    "owltron_power_fpr": row["owltron_power_fpr"],
                    "macro_benign_pcap_fpr": row["macro_benign_pcap_fpr"],
                }
            )
            reachability.append(
                {
                    "model_id": model_id,
                    "fpr_target": float(target),
                    "target_reached": bool(reached),
                    "benign_fpr": row["benign_fpr"],
                    "threshold": row["threshold"],
                }
            )

        ranking_rows = [r for r in ranking_rows if r.get("model_id") != model_id]
        ranking_rows.append(
            {
                "model_id": model_id,
                "roc_auc": art["roc_auc"],
                "pr_auc": art["pr_auc"],
                "accuracy_at_0_5": art["metrics"]["global"].get("accuracy"),
                "f1_at_0_5": art["metrics"]["global"].get("f1"),
                "benign_fpr_at_0_5": art["metrics"]["global"].get("benign_fpr"),
                "selection_role": "secondary",
                "note": "secondary metrics only; do not auto-select winner",
            }
        )
        model_summaries.append(
            {
                "model_id": model_id,
                "model_artifact_sha256": model_sha,
                "fit_seconds": fit_seconds,
                "score_seconds": score_seconds,
                "validation_rows_scored": EXPECTED_VAL_ROWS,
                "validation_pcaps_scored": EXPECTED_VAL_PCAPS,
                "test_pcaps_read": 0,
                "early_stopping_on_validation": False,
            }
        )

    comparison_rows = _parent_low_fpr_rows(parent_low) + new_low_rows

    _atomic_csv(
        out / "comparison_fixed_thresholds.csv",
        all_fixed,
        list(BAKEOFF_SWEEP_COLUMNS) + ["target_reached"],
    )
    _atomic_csv(
        out / "comparison_low_fpr.csv",
        comparison_rows,
        list(COMPARISON_COLUMNS),
    )
    _atomic_csv(
        out / "comparison_ranking_metrics.csv",
        ranking_rows,
        [
            "model_id",
            "roc_auc",
            "pr_auc",
            "accuracy_at_0_5",
            "f1_at_0_5",
            "benign_fpr_at_0_5",
            "selection_role",
            "note",
        ],
    )

    complete = {
        "status": "passed",
        "strategy_version": EXTERNAL_BOOST_VERSION,
        "parent_strategy": MODEL_FAMILY_VERSION,
        "purpose": "external_boosting_challengers",
        "feature_selection_status": "deferred",
        "feature_count": 27,
        "final_feature_count": "unresolved",
        "class_weights": "none",
        "hyperparameter_search": False,
        "early_stopping_on_validation": False,
        "threshold_selection": False,
        "fit": {
            "pcaps": EXPECTED_FIT_PCAPS,
            "rows": EXPECTED_FIT_ROWS,
            "attack_rows": EXPECTED_FIT_ATTACK,
            "benign_rows": EXPECTED_FIT_BENIGN,
        },
        "validation": {
            "pcaps": EXPECTED_VAL_PCAPS,
            "rows": EXPECTED_VAL_ROWS,
            "sampling": "never",
            "rows_scored_per_model": EXPECTED_VAL_ROWS,
            "pcaps_scored_per_model": EXPECTED_VAL_PCAPS,
        },
        "test": {"access": False, "pcaps_read": 0},
        "models": model_summaries,
        "low_fpr_targets": list(LOW_FPR_TARGETS),
        "target_reachability": reachability,
        "decision_rule": list(DECISION_RULE),
        "auto_declare_winner": False,
        "winner": None,
        "versions": _package_versions(),
        "artifacts": {
            "external_boost_contract": to_repo_relative(
                out / "external_boost_contract.json", project_root=root
            ),
            "comparison_low_fpr": to_repo_relative(
                out / "comparison_low_fpr.csv", project_root=root
            ),
            "comparison_fixed_thresholds": to_repo_relative(
                out / "comparison_fixed_thresholds.csv", project_root=root
            ),
            "comparison_ranking_metrics": to_repo_relative(
                out / "comparison_ranking_metrics.csv", project_root=root
            ),
            "output_dir": to_repo_relative(out, project_root=root),
        },
        "next": (
            "Review comparison_low_fpr.csv at ≤0.5% / ≤0.1% FPR using the "
            "predeclared ranking (min family → Recon → MQTT → macro → Owltron). "
            "Do not auto-replace HGB. Feature selection remains deferred. "
            "Do not touch TEST."
        ),
    }
    _assert_complete_gate(complete)
    _atomic_json(out / "external_boost_complete.json", complete)
    return complete


def format_prepare_external_boost_summary(payload: dict[str, Any]) -> str:
    return (
        "Phase 2B.4C — external boosting contract prepared\n"
        f"status: {payload.get('status')}\n"
        f"strategy_version: {payload.get('strategy_version')}\n"
        f"feature_selection_status: {payload.get('feature_selection_status')}\n"
        f"contract: {payload.get('contract_path')}\n"
        f"next: {payload.get('next')}\n"
    )


def format_external_boost_summary(payload: dict[str, Any]) -> str:
    lines = [
        "Phase 2B.4C — external boosting challengers",
        f"status: {payload.get('status')}",
        f"strategy_version: {payload.get('strategy_version')}",
        f"early_stopping_on_validation: {payload.get('early_stopping_on_validation')}",
        f"feature_selection_status: {payload.get('feature_selection_status')}",
        f"auto_declare_winner: {payload.get('auto_declare_winner')}",
        f"winner: {payload.get('winner')}",
    ]
    for m in payload.get("models") or []:
        lines.append(
            f"  {m['model_id']}: sha={str(m['model_artifact_sha256'])[:12]}… "
            f"fit_s={float(m['fit_seconds']):.1f} "
            f"score_s={float(m['score_seconds']):.1f} "
            f"rows={m['validation_rows_scored']} "
            f"pcaps={m['validation_pcaps_scored']} test={m['test_pcaps_read']}"
        )
    for pt in payload.get("target_reachability") or []:
        flag = "reached" if pt.get("target_reached") else "UNREACHABLE"
        lines.append(
            f"  {pt['model_id']} FPR≤{float(pt['fpr_target']):.4%}: {flag} "
            f"actual={float(pt['benign_fpr']):.4%} thr={float(pt['threshold']):.6f}"
        )
    arts = payload.get("artifacts") or {}
    if arts.get("comparison_low_fpr"):
        lines.append(f"comparison: {arts['comparison_low_fpr']}")
    if payload.get("next"):
        lines.append(f"next: {payload['next']}")
    return "\n".join(lines) + "\n"
