"""Phase 2B.4D: 22-feature rematch — HGB-C vs XGBoost vs CatBoost.

Uses the deferred nontemporal model-input set (22 features). HGB reuses the
Phase 2B.3B C_22_unweighted artifact. XGBoost/CatBoost use the same fixed
untuned configs as Phase 2B.4C (no early stopping). TEST stays sealed.
"""

from __future__ import annotations

import csv
import json
import platform
import shutil
import time
from pathlib import Path
from typing import Any, TextIO

import joblib
import numpy as np

from iot_pcap_pipeline.features.parquet import feature_schema_sha256
from iot_pcap_pipeline.features.schema import DEFAULT_FEATURE_SCHEMA_PATH, V1_FEATURE_NAMES
from iot_pcap_pipeline.modeling.baselines.ablations import (
    ABLATION_VERSION,
    DEFAULT_ABLATION_ROOT,
)
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
from iot_pcap_pipeline.modeling.baselines.extratrees import COMPARISON_COLUMNS
from iot_pcap_pipeline.modeling.baselines.model_family import (
    BAKEOFF_SWEEP_COLUMNS,
    LOW_FPR_TARGETS,
    _load_score_cache,
    _save_score_cache,
    _write_family_artifacts,
    metrics_at_threshold_bakeoff,
    score_bakeoff_tape,
)
from iot_pcap_pipeline.modeling.baselines.model_input import (
    DROPPED_TEMPORAL_FEATURES,
    V1_MODEL_INPUT_FEATURES,
    V1_MODEL_INPUT_VERSION,
)
from iot_pcap_pipeline.modeling.baselines.models import (
    CATBOOST_PARAMS,
    HGB_PARAMS,
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

FEATURE22_BOOST_VERSION = "phase2b4d_v1"
DEFAULT_FEATURE22_BOOST_ROOT = (
    DEFAULT_MODELING_DIR / "v1" / "baselines" / FEATURE22_BOOST_VERSION
)

HGB22_SOURCE_JOBLIB = (
    DEFAULT_ABLATION_ROOT / "models" / "C_22_unweighted.joblib"
)

FEATURES_22 = list(V1_MODEL_INPUT_FEATURES)
assert len(FEATURES_22) == 22

DECISION_RULE: list[str] = [
    "Compare only under the 22 nontemporal model-input features.",
    "At matched low FPR rank by: min attack-family recall → Recon → MQTT → "
    "macro → Owltron FPR.",
    "Practical band: ≤0.5% and ≤0.1% benign FPR.",
    "No hyperparameter search; no early stopping on TRAIN-validation.",
    "Do not auto-declare a winner.",
]

CANDIDATE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "model_id": "hgb_22",
        "display_name": "HGB 22 features (reused 2B.3B C)",
        "reuse_hgb22": True,
        "builder": None,
        "hyperparameters": dict(HGB_PARAMS),
    },
    {
        "model_id": "xgboost_22",
        "display_name": "XGBClassifier 22 features (fixed hist)",
        "reuse_hgb22": False,
        "builder": build_xgboost,
        "hyperparameters": dict(XGBOOST_PARAMS),
    },
    {
        "model_id": "catboost_22",
        "display_name": "CatBoostClassifier 22 features (fixed Logloss)",
        "reuse_hgb22": False,
        "builder": build_catboost,
        "hyperparameters": dict(CATBOOST_PARAMS),
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


def _select_feature_matrix(X_full: np.ndarray) -> np.ndarray:
    idxs = [list(V1_FEATURE_NAMES).index(name) for name in FEATURES_22]
    return X_full[:, idxs]


def require_hgb22_ready(*, project_root: Path) -> Path:
    src = project_root / HGB22_SOURCE_JOBLIB
    if not src.is_file():
        raise FeatureExtractionError(
            f"missing HGB-22 artifact: {src}. Run run-hgb-ablations first."
        )
    complete = project_root / DEFAULT_ABLATION_ROOT / "ablation_complete.json"
    if not complete.is_file():
        raise FeatureExtractionError(
            f"missing 2B.3B complete marker: {complete}"
        )
    payload = json.loads(complete.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise FeatureExtractionError("Phase 2B.3B ablation not passed")
    return src


def build_feature22_boost_contract(
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = (project_root or PROJECT_ROOT).resolve()
    require_fit_view_ready(project_root=root, smoke_only=False)
    hgb_src = require_hgb22_ready(project_root=root)
    hgb_sha = file_sha256(hgb_src)

    fit_man = root / DEFAULT_FIT_VIEW_MANIFEST_PATH
    train_contract = root / DEFAULT_TRAINING_VIEW_CONTRACT_PATH
    split_path = root / DEFAULT_SPLIT_MANIFEST_PATH
    schema_path = root / DEFAULT_FEATURE_SCHEMA_PATH

    return {
        "strategy_version": FEATURE22_BOOST_VERSION,
        "parent_ablation": ABLATION_VERSION,
        "purpose": "feature22_boost_rematch",
        "status": "frozen",
        "model_input_version": V1_MODEL_INPUT_VERSION,
        "feature_selection_status": "22_nontemporal_under_test",
        "feature_count": 22,
        "parent_feature_count": 27,
        "feature_names": list(FEATURES_22),
        "excluded_temporal_features": list(DROPPED_TEMPORAL_FEATURES),
        "label_mapping": dict(LABEL_MAPPING),
        "class_weights": "none",
        "hyperparameter_search": False,
        "early_stopping_on_validation": False,
        "threshold_selection": False,
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
            "hgb_22": {
                "reuse_phase2b3b_c": True,
                "source_artifact": to_repo_relative(hgb_src, project_root=root),
                "source_artifact_sha256": hgb_sha,
                "hyperparameters": dict(HGB_PARAMS),
            },
            "xgboost_22": {
                "hyperparameters": dict(XGBOOST_PARAMS),
                "early_stopping": False,
            },
            "catboost_22": {
                "hyperparameters": dict(CATBOOST_PARAMS),
                "early_stopping": False,
            },
        },
        "low_fpr_targets": list(LOW_FPR_TARGETS),
        "fixed_thresholds": list(FIXED_THRESHOLDS),
        "decision_rule": list(DECISION_RULE),
        "auto_declare_winner": False,
        "pins": {
            "feature_schema_sha256": feature_schema_sha256(schema_path),
            "training_view_contract_sha256": file_sha256(train_contract),
            "fit_view_manifest_sha256": file_sha256(fit_man),
            "modeling_split_manifest_sha256": file_sha256(split_path),
            "hgb22_model_artifact_sha256": hgb_sha,
        },
        "artifacts": {
            "feature_schema": to_repo_relative(schema_path, project_root=root),
            "training_view_contract": to_repo_relative(
                train_contract, project_root=root
            ),
            "fit_view_manifest": to_repo_relative(fit_man, project_root=root),
            "modeling_split_manifest": to_repo_relative(split_path, project_root=root),
            "hgb22_source_artifact": to_repo_relative(hgb_src, project_root=root),
        },
        "scope_limits": [
            "Exactly HGB-22, XGBoost-22, CatBoost-22.",
            "Pipeline still stores 27 features; model input is 22.",
            "No TEST access.",
            "No auto-winner.",
        ],
    }


def prepare_feature22_boost_rematch(
    *,
    project_root: Path | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    root = (project_root or PROJECT_ROOT).resolve()
    out = Path(output_dir or DEFAULT_FEATURE22_BOOST_ROOT)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    (out / "feature22_boost_complete.json").unlink(missing_ok=True)
    payload = build_feature22_boost_contract(project_root=root)
    contract_path = out / "feature22_boost_contract.json"
    _atomic_json(contract_path, payload)
    return {
        "status": "prepared",
        "strategy_version": FEATURE22_BOOST_VERSION,
        "contract_path": to_repo_relative(contract_path, project_root=root),
        "feature_count": 22,
        "next": (
            "Review feature22_boost_contract.json, then run "
            "`iot-pcap-pipeline run-feature22-boost-rematch`."
        ),
    }


def load_feature22_boost_contract(
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
            f"feature22_boost_contract.json missing: {p}. "
            "Run prepare-feature22-boost-rematch first."
        )
    payload = json.loads(p.read_text(encoding="utf-8"))
    if payload.get("strategy_version") != FEATURE22_BOOST_VERSION:
        raise FeatureExtractionError(
            f"unexpected strategy_version: {payload.get('strategy_version')!r}"
        )
    if int(payload.get("feature_count") or 0) != 22:
        raise FeatureExtractionError("feature_count must be 22")
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
    if int(val.get("rows_scored_per_model") or -1) != EXPECTED_VAL_ROWS:
        issues.append("wrong validation totals")
    if int(val.get("pcaps_scored_per_model") or -1) != EXPECTED_VAL_PCAPS:
        issues.append("wrong validation PCAPs")
    if int(payload.get("feature_count") or -1) != 22:
        issues.append("feature_count != 22")
    if test.get("access") is not False or int(test.get("pcaps_read", -1)) != 0:
        issues.append("TEST access")
    ids = [m.get("model_id") for m in (payload.get("models") or [])]
    if ids != ["hgb_22", "xgboost_22", "catboost_22"]:
        issues.append("expected hgb_22, xgboost_22, catboost_22")
    for m in payload.get("models") or []:
        if len(str(m.get("model_artifact_sha256") or "")) != 64:
            issues.append(f"missing hash for {m.get('model_id')}")
    arts = payload.get("artifacts") or {}
    if not arts.get("comparison_low_fpr"):
        issues.append("missing comparison_low_fpr")
    if issues:
        raise FeatureExtractionError(
            "feature22_boost_complete acceptance gate failed: " + "; ".join(issues)
        )


def run_feature22_boost_rematch(
    *,
    project_root: Path | None = None,
    output_dir: Path | str | None = None,
    fit_manifest_path: Path | str | None = None,
    split_manifest_path: Path | str | None = None,
    progress_file: TextIO | None = None,
    cache_scores: bool = True,
) -> dict[str, Any]:
    root = (project_root or PROJECT_ROOT).resolve()
    out = Path(output_dir or DEFAULT_FEATURE22_BOOST_ROOT)
    if not out.is_absolute():
        out = root / out

    contract = load_feature22_boost_contract(
        out / "feature22_boost_contract.json", project_root=root
    )
    pinned_hgb = str(
        (contract.get("pins") or {}).get("hgb22_model_artifact_sha256") or ""
    )
    require_fit_view_ready(project_root=root, smoke_only=False)
    hgb_src = require_hgb22_ready(project_root=root)

    fit_man = Path(fit_manifest_path or DEFAULT_FIT_VIEW_MANIFEST_PATH)
    if not fit_man.is_absolute():
        fit_man = root / fit_man
    split_path = Path(split_manifest_path or DEFAULT_SPLIT_MANIFEST_PATH)
    if not split_path.is_absolute():
        split_path = root / split_path

    out.mkdir(parents=True, exist_ok=True)
    (out / "feature22_boost_complete.json").unlink(missing_ok=True)
    models_dir = out / "models"
    pred_dir = out / "predictions"
    models_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    if progress_file is not None:
        progress_file.write("Loading FIT arrays and selecting 22 features...\n")
        progress_file.flush()
    fit = load_fit_arrays(fit_man, project_root=root, smoke_only=False)
    if fit.n_rows != EXPECTED_FIT_ROWS:
        raise FeatureExtractionError(f"FIT rows {fit.n_rows} != {EXPECTED_FIT_ROWS}")
    if fit.X.shape[1] != 27:
        raise FeatureExtractionError(f"expected FIT X ncols=27, got {fit.X.shape[1]}")
    X22 = _select_feature_matrix(fit.X)
    if X22.shape != (EXPECTED_FIT_ROWS, 22):
        raise FeatureExtractionError(f"X22 shape {X22.shape} != ({EXPECTED_FIT_ROWS}, 22)")
    if not np.isfinite(X22).all():
        raise FeatureExtractionError("non-finite values in 22-feature FIT matrix")

    all_fixed: list[dict[str, Any]] = []
    all_low: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    model_summaries: list[dict[str, Any]] = []
    reachability: list[dict[str, Any]] = []

    for spec in CANDIDATE_SPECS:
        model_id = str(spec["model_id"])
        model_dir = out / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{model_id}.joblib"
        score_cache = pred_dir / f"{model_id}_val_scores.npz"

        if progress_file is not None:
            progress_file.write(f"\n=== {model_id} ===\n")
            progress_file.flush()

        fit_seconds = 0.0
        reused = bool(spec["reuse_hgb22"])
        if reused:
            src_sha = file_sha256(hgb_src)
            if pinned_hgb and src_sha != pinned_hgb:
                raise FeatureExtractionError(
                    f"HGB-22 SHA drift: now={src_sha} pinned={pinned_hgb}"
                )
            shutil.copy2(hgb_src, model_path)
            model_sha = file_sha256(model_path)
            estimator = joblib.load(model_path)
            if progress_file is not None:
                progress_file.write(f"  reused 2B.3B C sha={model_sha[:12]}...\n")
                progress_file.flush()
        else:
            if progress_file is not None:
                progress_file.write(
                    f"  fitting {model_id} on 22 features (no early stopping)...\n"
                )
                progress_file.flush()
            builder = spec["builder"]
            assert builder is not None
            estimator = builder()
            t0 = time.perf_counter()
            estimator.fit(X22, fit.y)
            fit_seconds = time.perf_counter() - t0
            joblib.dump(estimator, model_path)
            model_sha = file_sha256(model_path)

        _ = attack_score_from_estimator(estimator, X22[: min(16, X22.shape[0])])

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
                feature_names=FEATURES_22,
                progress_file=progress_file,
            )
            score_seconds = time.perf_counter() - t1
            if cache_scores:
                _save_score_cache(score_cache, tape)

        if tape.n_rows != EXPECTED_VAL_ROWS:
            raise FeatureExtractionError(
                f"{model_id}: rows={tape.n_rows} != {EXPECTED_VAL_ROWS}"
            )
        if len(tape.pcap_table) != EXPECTED_VAL_PCAPS:
            raise FeatureExtractionError(
                f"{model_id}: pcaps={len(tape.pcap_table)} != {EXPECTED_VAL_PCAPS}"
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
                "feature_count": 22,
                "feature_names": list(FEATURES_22),
                "model_input_version": V1_MODEL_INPUT_VERSION,
                "early_stopping_on_validation": False,
                "reused_phase2b3b_c": reused,
                "versions": _package_versions(),
            },
            reused=reused,
            strategy_version=FEATURE22_BOOST_VERSION,
        )
        # Fix metadata feature_count (helper writes 27 by default).
        meta_path = model_dir / "model_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["feature_count"] = 22
        meta["feature_names"] = list(FEATURES_22)
        meta["bakeoff_feature_set"] = "nontemporal_22"
        _atomic_json(meta_path, meta)

        metrics_path = model_dir / "metrics.json"
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics_payload["selection_role"] = "secondary"
        metrics_payload["feature_count"] = 22
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
            low = {
                "model_id": model_id,
                "fpr_target": row["fpr_target"],
                "target_reached": bool(reached),
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
            all_low.append(low)
            reachability.append(
                {
                    "model_id": model_id,
                    "fpr_target": float(target),
                    "target_reached": bool(reached),
                    "benign_fpr": row["benign_fpr"],
                    "threshold": row["threshold"],
                    "recon_os_scan_recall": row["recon_os_scan_recall"],
                    "mqtt_publish_recall": row["mqtt_publish_recall"],
                    "min_attack_group_recall": row["min_attack_group_recall"],
                }
            )

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
                "reused_phase2b3b_c": reused,
                "early_stopping_on_validation": False,
            }
        )

    _atomic_csv(
        out / "comparison_fixed_thresholds.csv",
        all_fixed,
        list(BAKEOFF_SWEEP_COLUMNS) + ["target_reached"],
    )
    _atomic_csv(out / "comparison_low_fpr.csv", all_low, list(COMPARISON_COLUMNS))
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
        "strategy_version": FEATURE22_BOOST_VERSION,
        "purpose": "feature22_boost_rematch",
        "model_input_version": V1_MODEL_INPUT_VERSION,
        "feature_count": 22,
        "feature_names": list(FEATURES_22),
        "excluded_temporal_features": list(DROPPED_TEMPORAL_FEATURES),
        "class_weights": "none",
        "hyperparameter_search": False,
        "early_stopping_on_validation": False,
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
            "feature22_boost_contract": to_repo_relative(
                out / "feature22_boost_contract.json", project_root=root
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
            "Review comparison_low_fpr.csv for hgb_22 / xgboost_22 / catboost_22 "
            "at ≤0.5% and ≤0.1% FPR. Do not auto-declare a winner. TEST sealed."
        ),
    }
    _assert_complete_gate(complete)
    _atomic_json(out / "feature22_boost_complete.json", complete)
    return complete


def format_prepare_feature22_boost_summary(payload: dict[str, Any]) -> str:
    return (
        "Phase 2B.4D — 22-feature boost rematch contract prepared\n"
        f"status: {payload.get('status')}\n"
        f"strategy_version: {payload.get('strategy_version')}\n"
        f"feature_count: {payload.get('feature_count')}\n"
        f"contract: {payload.get('contract_path')}\n"
        f"next: {payload.get('next')}\n"
    )


def format_feature22_boost_summary(payload: dict[str, Any]) -> str:
    lines = [
        "Phase 2B.4D — 22-feature boost rematch",
        f"status: {payload.get('status')}",
        f"strategy_version: {payload.get('strategy_version')}",
        f"feature_count: {payload.get('feature_count')}",
        f"auto_declare_winner: {payload.get('auto_declare_winner')}",
        f"winner: {payload.get('winner')}",
    ]
    for m in payload.get("models") or []:
        lines.append(
            f"  {m['model_id']}: sha={str(m['model_artifact_sha256'])[:12]}… "
            f"fit_s={float(m['fit_seconds']):.1f} "
            f"score_s={float(m['score_seconds']):.1f}"
        )
    for pt in payload.get("target_reachability") or []:
        if float(pt["fpr_target"]) not in (0.005, 0.001):
            continue
        flag = "ok" if pt.get("target_reached") else "soft"
        lines.append(
            f"  {pt['model_id']} ≤{float(pt['fpr_target']):.1%}: {flag} "
            f"fpr={float(pt['benign_fpr']):.3%} "
            f"recon={float(pt['recon_os_scan_recall']):.2%} "
            f"mqtt={float(pt['mqtt_publish_recall']):.2%} "
            f"min={float(pt['min_attack_group_recall']):.2%}"
        )
    arts = payload.get("artifacts") or {}
    if arts.get("comparison_low_fpr"):
        lines.append(f"comparison: {arts['comparison_low_fpr']}")
    if payload.get("next"):
        lines.append(f"next: {payload['next']}")
    return "\n".join(lines) + "\n"
