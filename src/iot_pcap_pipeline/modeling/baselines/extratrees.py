"""Phase 2B.4B: ExtraTrees final sklearn model-family challenger.

Reuses the Phase 2B.4 data contract (27 features, deferred feature selection).
Does not auto-replace HGB; stops for review after the four-family low-FPR table.
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
from iot_pcap_pipeline.modeling.baselines.data import (
    assert_feature_columns,
    load_fit_arrays,
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
    EXTRA_TREES_PARAMS,
    RANDOM_SEED,
    attack_score_from_estimator,
    build_extra_trees,
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

EXTRA_TREES_VERSION = "phase2b4b_v1"
DEFAULT_EXTRA_TREES_ROOT = (
    DEFAULT_MODELING_DIR / "v1" / "baselines" / EXTRA_TREES_VERSION
)
DEFAULT_EXTRA_TREES_CONTRACT_PATH = (
    DEFAULT_EXTRA_TREES_ROOT / "extratrees_contract.json"
)

MODEL_ID = "extratrees"

COMPARISON_COLUMNS: tuple[str, ...] = (
    "model_id",
    "fpr_target",
    "target_reached",
    "threshold",
    "benign_fp",
    "benign_support",
    "benign_fpr",
    "ddos_tcp_recall",
    "dos_tcp_recall",
    "mqtt_publish_recall",
    "recon_os_scan_recall",
    "macro_attack_group_recall",
    "min_attack_group_recall",
    "owltron_interaction_fpr",
    "profiling_idle_fpr",
    "owltron_power_fpr",
    "macro_benign_pcap_fpr",
)

DECISION_RULE: list[str] = [
    "ExtraTrees advances only if it materially improves the HGB frontier.",
    "Primary ranking: (1) min attack-group recall, (2) Recon, (3) MQTT, "
    "(4) macro attack-group recall, (5) Owltron interaction FPR, (6) DDoS/DoS.",
    "Practical comparison band: ≤0.5% and ≤0.1% benign FPR.",
    "Do not replace HGB for trivial flood-recall gains.",
    "Useful challenger: same/lower benign FPR AND meaningfully higher Recon "
    "AND no meaningful MQTT regression; ties prefer HGB.",
]


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
    import numpy
    import pyarrow
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
        "pyarrow": pyarrow.__version__,
        "platform": platform.platform(),
    }


def assert_fit_arrays_ready(fit: Any) -> None:
    """Hard FIT checks before ExtraTrees fit."""
    if fit.X.shape != (EXPECTED_FIT_ROWS, len(V1_FEATURE_NAMES)):
        raise FeatureExtractionError(
            f"X_fit.shape={fit.X.shape} != ({EXPECTED_FIT_ROWS}, "
            f"{len(V1_FEATURE_NAMES)})"
        )
    if fit.n_attack != EXPECTED_FIT_ATTACK or fit.n_benign != EXPECTED_FIT_BENIGN:
        raise FeatureExtractionError(
            f"FIT labels attack={fit.n_attack} benign={fit.n_benign}; "
            f"expected {EXPECTED_FIT_ATTACK}/{EXPECTED_FIT_BENIGN}"
        )
    if not np.isfinite(fit.X).all():
        raise FeatureExtractionError("FIT features contain non-finite values")
    assert_feature_columns(list(V1_FEATURE_NAMES))


def require_phase2b4_complete(
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = (project_root or PROJECT_ROOT).resolve()
    path = root / DEFAULT_MODEL_FAMILY_ROOT / "model_family_complete.json"
    if not path.is_file():
        raise FeatureExtractionError(
            f"Phase 2B.4 complete marker missing: {path}. "
            "Run run-model-family-bakeoff first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise FeatureExtractionError(
            f"Phase 2B.4 not passed: status={payload.get('status')!r}"
        )
    if payload.get("strategy_version") != MODEL_FAMILY_VERSION:
        raise FeatureExtractionError(
            f"unexpected parent strategy_version: {payload.get('strategy_version')!r}"
        )
    return payload


def build_extratrees_contract(
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = (project_root or PROJECT_ROOT).resolve()
    require_fit_view_ready(project_root=root, smoke_only=False)
    parent = require_phase2b4_complete(project_root=root)
    parent_contract_path = root / DEFAULT_MODEL_FAMILY_ROOT / "model_family_contract.json"
    parent_contract = json.loads(parent_contract_path.read_text(encoding="utf-8"))

    fit_man = root / DEFAULT_FIT_VIEW_MANIFEST_PATH
    train_contract = root / DEFAULT_TRAINING_VIEW_CONTRACT_PATH
    split_path = root / DEFAULT_SPLIT_MANIFEST_PATH
    schema_path = root / DEFAULT_FEATURE_SCHEMA_PATH

    pins = dict(parent_contract.get("pins") or {})
    # Recompute and verify match with parent pins.
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

    return {
        "strategy_version": EXTRA_TREES_VERSION,
        "parent_strategy": MODEL_FAMILY_VERSION,
        "purpose": "final_sklearn_model_family_challenger",
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
        "model": {
            "model_id": MODEL_ID,
            "family": "ExtraTreesClassifier",
            "hyperparameters": dict(EXTRA_TREES_PARAMS),
        },
        "low_fpr_targets": list(LOW_FPR_TARGETS),
        "fixed_thresholds": list(FIXED_THRESHOLDS),
        "ranking_criteria": list(RANKING_CRITERIA),
        "decision_rule": list(DECISION_RULE),
        "auto_declare_winner": False,
        "auto_advance_extratrees": False,
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
            "parent_comparison_low_fpr": to_repo_relative(
                root / DEFAULT_MODEL_FAMILY_ROOT / "comparison_low_fpr.csv",
                project_root=root,
            ),
        },
        "scope_limits": [
            "Single ExtraTrees challenger only; no grids or additional families.",
            "Feature 22-vs-27 remains deferred.",
            "No TEST access.",
            "Do not auto-replace HGB.",
        ],
    }


def prepare_extratrees_challenger(
    *,
    project_root: Path | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    root = (project_root or PROJECT_ROOT).resolve()
    out = Path(output_dir or DEFAULT_EXTRA_TREES_ROOT)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    (out / "extratrees_complete.json").unlink(missing_ok=True)
    payload = build_extratrees_contract(project_root=root)
    contract_path = out / "extratrees_contract.json"
    _atomic_json(contract_path, payload)
    return {
        "status": "prepared",
        "strategy_version": EXTRA_TREES_VERSION,
        "contract_path": to_repo_relative(contract_path, project_root=root),
        "feature_selection_status": payload["feature_selection_status"],
        "next": (
            "Review extratrees_contract.json, then run "
            "`iot-pcap-pipeline run-extratrees-challenger`."
        ),
    }


def load_extratrees_contract(
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
            f"extratrees_contract.json missing: {p}. "
            "Run prepare-extratrees-challenger first."
        )
    payload = json.loads(p.read_text(encoding="utf-8"))
    if payload.get("strategy_version") != EXTRA_TREES_VERSION:
        raise FeatureExtractionError(
            f"unexpected extratrees strategy_version: "
            f"{payload.get('strategy_version')!r}"
        )
    if payload.get("parent_strategy") != MODEL_FAMILY_VERSION:
        raise FeatureExtractionError(
            f"parent_strategy must be {MODEL_FAMILY_VERSION}"
        )
    if payload.get("feature_selection_status") != "deferred":
        raise FeatureExtractionError("feature_selection_status must be deferred")
    if int(payload.get("feature_count") or 0) != 27:
        raise FeatureExtractionError("feature_count must be 27")
    if payload.get("test_access") is not False:
        raise FeatureExtractionError("test_access must be false")
    return payload


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parent_low_fpr_rows(parent_csv: Path) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for raw in _read_csv_rows(parent_csv):
        target = float(raw["fpr_target"]) if raw.get("fpr_target") not in ("", None) else None
        benign_fpr = (
            float(raw["benign_fpr"]) if raw.get("benign_fpr") not in ("", None) else None
        )
        reached = (
            target is not None
            and benign_fpr is not None
            and benign_fpr <= target + 1e-15
        )
        rows_out.append(
            {
                "model_id": raw["model_id"],
                "fpr_target": target,
                "target_reached": reached,
                "threshold": float(raw["threshold"]),
                "benign_fp": int(float(raw["benign_fp"])),
                "benign_support": int(float(raw["benign_support"])),
                "benign_fpr": benign_fpr,
                "ddos_tcp_recall": float(raw["ddos_tcp_recall"]),
                "dos_tcp_recall": float(raw["dos_tcp_recall"]),
                "mqtt_publish_recall": float(raw["mqtt_publish_recall"]),
                "recon_os_scan_recall": float(raw["recon_os_scan_recall"]),
                "macro_attack_group_recall": float(raw["macro_attack_group_recall"]),
                "min_attack_group_recall": float(raw["min_attack_group_recall"]),
                "owltron_interaction_fpr": float(raw["owltron_interaction_fpr"]),
                "profiling_idle_fpr": float(raw["profiling_idle_fpr"]),
                "owltron_power_fpr": float(raw["owltron_power_fpr"]),
                "macro_benign_pcap_fpr": float(raw.get("macro_benign_pcap_fpr") or 0.0)
                if raw.get("macro_benign_pcap_fpr") not in ("", None)
                else None,
            }
        )
    return rows_out


def _assert_complete_gate(payload: dict[str, Any]) -> None:
    """Refuse a passed marker unless acceptance checks hold."""
    issues: list[str] = []
    fit = payload.get("fit") or {}
    val = payload.get("validation") or {}
    test = payload.get("test") or {}
    model = payload.get("model") or {}
    if int(fit.get("rows") or -1) != EXPECTED_FIT_ROWS:
        issues.append("wrong FIT totals")
    if int(fit.get("pcaps") or -1) != EXPECTED_FIT_PCAPS:
        issues.append("wrong FIT PCAPs")
    if int(val.get("rows_scored") or -1) != EXPECTED_VAL_ROWS:
        issues.append("wrong validation totals")
    if int(val.get("pcaps_scored") or -1) != EXPECTED_VAL_PCAPS:
        issues.append("wrong validation PCAPs")
    if val.get("sampling") != "never":
        issues.append("validation must be unsampled")
    if int(payload.get("feature_count") or -1) != 27:
        issues.append("feature_count != 27")
    if test.get("access") is not False or int(test.get("pcaps_read", -1)) != 0:
        issues.append("TEST access")
    sha = str(model.get("model_artifact_sha256") or "")
    if len(sha) != 64:
        issues.append("missing model hash")
    if model.get("family") != "ExtraTreesClassifier":
        issues.append("model family mismatch")
    if int(model.get("n_estimators") or -1) != 100:
        issues.append("n_estimators != 100")
    if model.get("class_weight") is not None:
        issues.append("class_weight must be none")
    if int(model.get("random_state") or -1) != RANDOM_SEED:
        issues.append("random_state != 42")
    arts = payload.get("artifacts") or {}
    for key in (
        "comparison_low_fpr",
        "comparison_fixed_thresholds",
        "comparison_ranking_metrics",
        "extratrees_metrics",
        "extratrees_pcap_metrics",
    ):
        if not arts.get(key):
            issues.append(f"missing artifact {key}")
    if issues:
        raise FeatureExtractionError(
            "extratrees_complete acceptance gate failed: " + "; ".join(issues)
        )


def run_extratrees_challenger(
    *,
    project_root: Path | None = None,
    output_dir: Path | str | None = None,
    fit_manifest_path: Path | str | None = None,
    split_manifest_path: Path | str | None = None,
    progress_file: TextIO | None = None,
    cache_scores: bool = True,
) -> dict[str, Any]:
    """Fit ExtraTrees once, score full TRAIN-validation, compare to 2B.4 families."""
    root = (project_root or PROJECT_ROOT).resolve()
    out = Path(output_dir or DEFAULT_EXTRA_TREES_ROOT)
    if not out.is_absolute():
        out = root / out

    contract = load_extratrees_contract(
        out / "extratrees_contract.json", project_root=root
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
    (out / "extratrees_complete.json").unlink(missing_ok=True)
    models_dir = out / "models"
    pred_dir = out / "predictions"
    model_dir = out / MODEL_ID
    models_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    if progress_file is not None:
        progress_file.write("Loading FIT arrays (27 features)...\n")
        progress_file.flush()
    fit = load_fit_arrays(fit_man, project_root=root, smoke_only=False)
    assert_fit_arrays_ready(fit)

    model_path = models_dir / f"{MODEL_ID}.joblib"
    score_cache = pred_dir / f"{MODEL_ID}_val_scores.npz"

    if progress_file is not None:
        progress_file.write("Fitting ExtraTreesClassifier...\n")
        progress_file.flush()
    estimator = build_extra_trees()
    t0 = time.perf_counter()
    estimator.fit(fit.X, fit.y)
    fit_seconds = time.perf_counter() - t0
    joblib.dump(estimator, model_path)
    model_sha = file_sha256(model_path)

    # Sanity: positive-class predict_proba column.
    _ = attack_score_from_estimator(estimator, fit.X[: min(16, fit.n_rows)])

    if cache_scores and score_cache.is_file():
        if progress_file is not None:
            progress_file.write(f"  loading score cache {score_cache.name}\n")
            progress_file.flush()
        tape = _load_score_cache(score_cache)
        score_seconds = 0.0
    else:
        if progress_file is not None:
            progress_file.write("Scoring TRAIN-validation for extratrees...\n")
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
            f"validation_rows_scored={tape.n_rows} != {EXPECTED_VAL_ROWS}"
        )
    if len(tape.pcap_table) != EXPECTED_VAL_PCAPS:
        raise FeatureExtractionError(
            f"validation_pcaps_scored={len(tape.pcap_table)} != {EXPECTED_VAL_PCAPS}"
        )

    art = _write_family_artifacts(
        model_dir=model_dir,
        model_id=MODEL_ID,
        display_name="ExtraTreesClassifier",
        hyperparameters=dict(EXTRA_TREES_PARAMS),
        model_path=model_path,
        model_sha=model_sha,
        fit_seconds=fit_seconds,
        score_seconds=score_seconds,
        tape=tape,
        project_root=root,
        extra_metadata={
            "parent_strategy": MODEL_FAMILY_VERSION,
            "purpose": "final_sklearn_model_family_challenger",
            "selection_role_at_0_5": "secondary",
            "versions": _package_versions(),
        },
        reused=False,
        strategy_version=EXTRA_TREES_VERSION,
    )
    # Annotate metrics.json selection role.
    metrics_path = model_dir / "metrics.json"
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics_payload["selection_role"] = "secondary"
    metrics_payload["note"] = (
        "Threshold-0.5 / ROC-AUC / PR-AUC are secondary continuity metrics; "
        "selection uses comparison_low_fpr.csv at ≤0.5% and ≤0.1% FPR."
    )
    _atomic_json(metrics_path, metrics_payload)

    # Fixed-threshold frontier (ExtraTrees only).
    fixed_rows = [
        metrics_at_threshold_bakeoff(
            tape,
            threshold=t,
            model_id=MODEL_ID,
            point_type="fixed_threshold",
        )
        for t in FIXED_THRESHOLDS
    ]
    for row in fixed_rows:
        row["target_reached"] = ""

    # Low-FPR ExtraTrees rows with reachability.
    et_low_rows: list[dict[str, Any]] = []
    for target in LOW_FPR_TARGETS:
        thr, reached = threshold_for_benign_fpr_with_reachability(
            tape.as_validation_tape(), target
        )
        row = metrics_at_threshold_bakeoff(
            tape,
            threshold=thr,
            model_id=MODEL_ID,
            point_type="fpr_target",
            fpr_target=target,
        )
        row["target_reached"] = bool(reached)
        et_low_rows.append(row)

    parent_low = root / DEFAULT_MODEL_FAMILY_ROOT / "comparison_low_fpr.csv"
    if not parent_low.is_file():
        raise FeatureExtractionError(f"missing parent low-FPR table: {parent_low}")
    four_family = _parent_low_fpr_rows(parent_low)
    for row in et_low_rows:
        four_family.append(
            {
                "model_id": MODEL_ID,
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

    # Ranking metrics: reuse 2B.4 + ExtraTrees secondary row.
    parent_rank = root / DEFAULT_MODEL_FAMILY_ROOT / "comparison_ranking_metrics.csv"
    ranking_rows = _read_csv_rows(parent_rank) if parent_rank.is_file() else []
    ranking_rows.append(
        {
            "model_id": MODEL_ID,
            "roc_auc": art["roc_auc"],
            "pr_auc": art["pr_auc"],
            "accuracy_at_0_5": art["metrics"]["global"].get("accuracy"),
            "f1_at_0_5": art["metrics"]["global"].get("f1"),
            "benign_fpr_at_0_5": art["metrics"]["global"].get("benign_fpr"),
            "selection_role": "secondary",
            "note": "secondary metrics only; do not auto-select winner",
        }
    )

    _atomic_csv(
        out / "comparison_fixed_thresholds.csv",
        fixed_rows,
        list(BAKEOFF_SWEEP_COLUMNS) + ["target_reached"],
    )
    _atomic_csv(out / "comparison_low_fpr.csv", four_family, list(COMPARISON_COLUMNS))
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
        "strategy_version": EXTRA_TREES_VERSION,
        "parent_strategy": MODEL_FAMILY_VERSION,
        "purpose": contract.get("purpose"),
        "feature_selection_status": "deferred",
        "feature_count": 27,
        "final_feature_count": "unresolved",
        "threshold_selection": False,
        "class_weights": "none",
        "fit": {
            "pcaps": EXPECTED_FIT_PCAPS,
            "rows": EXPECTED_FIT_ROWS,
            "attack_rows": EXPECTED_FIT_ATTACK,
            "benign_rows": EXPECTED_FIT_BENIGN,
        },
        "validation": {
            "pcaps": EXPECTED_VAL_PCAPS,
            "rows": EXPECTED_VAL_ROWS,
            "pcaps_inventory": EXPECTED_VAL_PCAPS,
            "pcaps_scored": EXPECTED_VAL_PCAPS,
            "rows_scored": EXPECTED_VAL_ROWS,
            "sampling": "never",
        },
        "test": {"access": False, "pcaps_read": 0},
        "model": {
            "model_id": MODEL_ID,
            "family": "ExtraTreesClassifier",
            "n_estimators": EXTRA_TREES_PARAMS["n_estimators"],
            "bootstrap": EXTRA_TREES_PARAMS["bootstrap"],
            "class_weight": EXTRA_TREES_PARAMS["class_weight"],
            "random_state": RANDOM_SEED,
            "n_jobs": EXTRA_TREES_PARAMS["n_jobs"],
            "model_artifact_sha256": model_sha,
            "fit_seconds": fit_seconds,
            "score_seconds": score_seconds,
            "versions": _package_versions(),
        },
        "low_fpr_targets": list(LOW_FPR_TARGETS),
        "extratrees_target_reachability": [
            {
                "fpr_target": float(r["fpr_target"]),
                "target_reached": bool(r["target_reached"]),
                "benign_fpr": r["benign_fpr"],
                "threshold": r["threshold"],
            }
            for r in et_low_rows
        ],
        "decision_rule": list(DECISION_RULE),
        "auto_declare_winner": False,
        "auto_advance_extratrees": False,
        "winner": None,
        "artifacts": {
            "extratrees_contract": to_repo_relative(
                out / "extratrees_contract.json", project_root=root
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
            "extratrees_metrics": to_repo_relative(
                model_dir / "metrics.json", project_root=root
            ),
            "extratrees_pcap_metrics": to_repo_relative(
                model_dir / "pcap_metrics.csv", project_root=root
            ),
            "extratrees_attack_group_metrics": to_repo_relative(
                model_dir / "attack_group_metrics.csv", project_root=root
            ),
            "extratrees_benign_group_metrics": to_repo_relative(
                model_dir / "benign_group_metrics.csv", project_root=root
            ),
            "output_dir": to_repo_relative(out, project_root=root),
        },
        "next": (
            "Review comparison_low_fpr.csv (HGB/AdaBoost/RF/ExtraTrees). "
            "Advance ExtraTrees only if it materially beats HGB at ≤0.5% / ≤0.1% "
            "FPR on Recon without MQTT regression; otherwise keep HGB. "
            "Feature selection remains deferred. Do not touch TEST."
        ),
    }
    _assert_complete_gate(complete)
    _atomic_json(out / "extratrees_complete.json", complete)
    return complete


def format_prepare_extratrees_summary(payload: dict[str, Any]) -> str:
    return (
        "Phase 2B.4B — ExtraTrees contract prepared\n"
        f"status: {payload.get('status')}\n"
        f"strategy_version: {payload.get('strategy_version')}\n"
        f"feature_selection_status: {payload.get('feature_selection_status')}\n"
        f"contract: {payload.get('contract_path')}\n"
        f"next: {payload.get('next')}\n"
    )


def format_extratrees_summary(payload: dict[str, Any]) -> str:
    lines = [
        "Phase 2B.4B — ExtraTrees final challenger",
        f"status: {payload.get('status')}",
        f"strategy_version: {payload.get('strategy_version')}",
        f"parent_strategy: {payload.get('parent_strategy')}",
        f"feature_selection_status: {payload.get('feature_selection_status')}",
        f"auto_advance_extratrees: {payload.get('auto_advance_extratrees')}",
        f"winner: {payload.get('winner')}",
    ]
    model = payload.get("model") or {}
    lines.append(
        f"model: {model.get('family')} n_estimators={model.get('n_estimators')} "
        f"bootstrap={model.get('bootstrap')} sha={str(model.get('model_artifact_sha256') or '')[:12]}… "
        f"fit_s={float(model.get('fit_seconds') or 0):.1f} "
        f"score_s={float(model.get('score_seconds') or 0):.1f}"
    )
    for pt in payload.get("extratrees_target_reachability") or []:
        flag = "reached" if pt.get("target_reached") else "UNREACHABLE"
        lines.append(
            f"  FPR≤{float(pt['fpr_target']):.4%}: {flag} "
            f"actual={float(pt['benign_fpr']):.4%} thr={float(pt['threshold']):.6f}"
        )
    arts = payload.get("artifacts") or {}
    if arts.get("comparison_low_fpr"):
        lines.append(f"four_family: {arts['comparison_low_fpr']}")
    if payload.get("next"):
        lines.append(f"next: {payload['next']}")
    return "\n".join(lines) + "\n"
