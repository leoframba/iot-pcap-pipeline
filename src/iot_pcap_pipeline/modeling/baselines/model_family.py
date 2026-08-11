"""Phase 2B.4: model-family bake-off (HGB vs AdaBoost vs Random Forest).

Feature selection is explicitly deferred: all candidates use all 27 V1 features
only to hold representation constant. TEST stays sealed.
"""

from __future__ import annotations

import csv
import json
import platform
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import joblib
import numpy as np

from iot_pcap_pipeline.features.schema import V1_FEATURE_NAMES
from iot_pcap_pipeline.modeling.baselines.constants import (
    ATTACK_VAL_GROUPS,
    DECISION_THRESHOLD,
    EXPECTED_FIT_ATTACK,
    EXPECTED_FIT_BENIGN,
    EXPECTED_FIT_PCAPS,
    EXPECTED_FIT_ROWS,
    EXPECTED_VAL_ATTACK,
    EXPECTED_VAL_BENIGN,
    EXPECTED_VAL_PCAPS,
    EXPECTED_VAL_ROWS,
    LABEL_MAPPING,
)
from iot_pcap_pipeline.modeling.baselines.contract import (
    DEFAULT_BASELINES_ROOT,
    load_frozen_baseline_contract,
    require_fit_view_ready,
)
from iot_pcap_pipeline.modeling.baselines.data import (
    iter_validation_batches,
    load_fit_arrays,
    load_validation_specs,
    validate_validation_inventory,
)
from iot_pcap_pipeline.modeling.baselines.metrics import (
    ConfusionCounts,
    GroupAccumulator,
    benign_group_key,
    global_ranking_metrics,
    macro_mean,
    metrics_from_confusion,
)
from iot_pcap_pipeline.modeling.baselines.models import (
    ADABOOST_IMPLEMENTATION_NOTE,
    ADABOOST_PAPER_REFERENCE,
    ADABOOST_PARAMS,
    HGB_PARAMS,
    RANDOM_FOREST_PARAMS,
    RANDOM_SEED,
    attack_score_from_estimator,
    build_adaboost,
    build_random_forest,
)
from iot_pcap_pipeline.modeling.baselines.threshold_sweep import (
    ATTACK_GROUP_CODES,
    BENIGN_GROUP_CODES,
    FIXED_THRESHOLDS,
    SWEEP_ROW_COLUMNS,
    ValidationScoreTape,
    metrics_at_threshold,
    require_phase2b2_complete,
    threshold_for_benign_fpr,
)
from iot_pcap_pipeline.modeling.freeze import FROZEN_SAMPLING_PLAN_ID
from iot_pcap_pipeline.modeling.view import (
    DEFAULT_FIT_VIEW_MANIFEST_PATH,
    DEFAULT_SPLIT_MANIFEST_PATH,
    DEFAULT_TRAINING_VIEW_CONTRACT_PATH,
    file_sha256,
)
from iot_pcap_pipeline.features.parquet import feature_schema_sha256
from iot_pcap_pipeline.features.schema import DEFAULT_FEATURE_SCHEMA_PATH
from iot_pcap_pipeline.paths import (
    DEFAULT_MODELING_DIR,
    PROJECT_ROOT,
    to_repo_relative,
)
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError

MODEL_FAMILY_VERSION = "phase2b4_v1"
DEFAULT_MODEL_FAMILY_ROOT = (
    DEFAULT_MODELING_DIR / "v1" / "baselines" / MODEL_FAMILY_VERSION
)
DEFAULT_MODEL_FAMILY_CONTRACT_PATH = (
    DEFAULT_MODEL_FAMILY_ROOT / "model_family_contract.json"
)

HGB_SOURCE_JOBLIB = (
    DEFAULT_BASELINES_ROOT
    / "hist_gradient_boosting"
    / "models"
    / "hist_gradient_boosting.joblib"
)

# Primary low-FPR comparison band for model-family selection.
LOW_FPR_TARGETS: tuple[float, ...] = (
    0.01,
    0.005,
    0.0025,
    0.001,
    0.0005,
)

# Predeclared ranking rule (recorded in contract; code does not auto-pick a winner).
RANKING_CRITERIA: list[str] = [
    "Reject candidates with a material deployment problem at low FPR.",
    "Then compare primarily by: (1) minimum attack-group recall, "
    "(2) Recon OS Scan recall, (3) MQTT held-out recall, "
    "(4) macro attack-group recall, (5) Owltron interaction FPR, "
    "(6) DDoS/DoS recall.",
    "ROC-AUC / PR-AUC / accuracy / global F1 are secondary and must not "
    "select the winner automatically.",
]

BAKEOFF_SWEEP_COLUMNS: tuple[str, ...] = SWEEP_ROW_COLUMNS + (
    "macro_benign_pcap_fpr",
)

CANDIDATE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "model_id": "hgb",
        "display_name": "HistGradientBoostingClassifier",
        "reuse_phase2b2": True,
        "builder": None,
        "hyperparameters": dict(HGB_PARAMS),
    },
    {
        "model_id": "adaboost",
        "display_name": "AdaBoostClassifier (paper-inspired)",
        "reuse_phase2b2": False,
        "builder": build_adaboost,
        "hyperparameters": dict(ADABOOST_PARAMS),
        "paper_reference": dict(ADABOOST_PAPER_REFERENCE),
        "implementation": dict(ADABOOST_IMPLEMENTATION_NOTE),
    },
    {
        "model_id": "random_forest",
        "display_name": "RandomForestClassifier (paper-inspired)",
        "reuse_phase2b2": False,
        "builder": build_random_forest,
        "hyperparameters": dict(RANDOM_FOREST_PARAMS),
    },
)


@dataclass
class BakeoffScoreTape:
    y_true: np.ndarray
    scores: np.ndarray
    group_code: np.ndarray
    pcap_code: np.ndarray
    pcap_table: list[dict[str, str]]

    @property
    def n_rows(self) -> int:
        return int(self.y_true.shape[0])

    def as_validation_tape(self) -> ValidationScoreTape:
        return ValidationScoreTape(
            y_true=self.y_true,
            scores=self.scores,
            group_code=self.group_code,
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


def build_model_family_contract(
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Freeze Phase 2B.4 inputs before any AdaBoost/RF training."""
    root = (project_root or PROJECT_ROOT).resolve()
    require_fit_view_ready(project_root=root, smoke_only=False)
    b2 = require_phase2b2_complete(project_root=root)
    _ = load_frozen_baseline_contract(
        DEFAULT_BASELINES_ROOT / "baseline_contract.json",
        project_root=root,
    )

    fit_man = root / DEFAULT_FIT_VIEW_MANIFEST_PATH
    train_contract = root / DEFAULT_TRAINING_VIEW_CONTRACT_PATH
    split_path = root / DEFAULT_SPLIT_MANIFEST_PATH
    schema_path = root / DEFAULT_FEATURE_SCHEMA_PATH
    hgb_src = root / HGB_SOURCE_JOBLIB
    if not hgb_src.is_file():
        raise FeatureExtractionError(f"missing reused HGB artifact: {hgb_src}")
    hgb_sha = file_sha256(hgb_src)
    meta_path = (
        DEFAULT_BASELINES_ROOT
        / "hist_gradient_boosting"
        / "model_metadata.json"
    )
    meta = json.loads((root / meta_path).read_text(encoding="utf-8"))
    expected_sha = str(meta.get("model_artifact_sha256") or "")
    if expected_sha and hgb_sha != expected_sha:
        raise FeatureExtractionError(
            f"HGB artifact SHA mismatch: file={hgb_sha} metadata={expected_sha}"
        )

    return {
        "strategy_version": MODEL_FAMILY_VERSION,
        "task": "model_family_bakeoff",
        "status": "frozen",
        "model_families": ["hgb", "adaboost", "random_forest"],
        "feature_selection_status": "deferred",
        "bakeoff_feature_set": "all_27_v1",
        "final_feature_count": "unresolved",
        "feature_selection_note": (
            "all_27 used only to hold the feature representation constant "
            "during model-family comparison; this is not the final "
            "feature-selection decision."
        ),
        "feature_count": len(V1_FEATURE_NAMES),
        "feature_names": list(V1_FEATURE_NAMES),
        "label_mapping": dict(LABEL_MAPPING),
        "class_weights": "none",
        "hyperparameter_search": False,
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
            "attack_rows": EXPECTED_VAL_ATTACK,
            "benign_rows": EXPECTED_VAL_BENIGN,
            "sampling": "never",
        },
        "test": {"access": False, "pcaps_read": 0},
        "low_fpr_targets": list(LOW_FPR_TARGETS),
        "fixed_thresholds": list(FIXED_THRESHOLDS),
        "ranking_criteria": list(RANKING_CRITERIA),
        "auto_declare_winner": False,
        "candidates": {
            "hgb": {
                "reuse_phase2b2": True,
                "hyperparameters": dict(HGB_PARAMS),
                "source_artifact": to_repo_relative(hgb_src, project_root=root),
                "source_artifact_sha256": hgb_sha,
            },
            "adaboost": {
                "reuse_phase2b2": False,
                "paper_inspired": True,
                "hyperparameters": dict(ADABOOST_PARAMS),
                "paper_reference": dict(ADABOOST_PAPER_REFERENCE),
                "implementation": dict(ADABOOST_IMPLEMENTATION_NOTE),
            },
            "random_forest": {
                "reuse_phase2b2": False,
                "paper_inspired": True,
                "hyperparameters": dict(RANDOM_FOREST_PARAMS),
            },
        },
        "pins": {
            "feature_schema_sha256": feature_schema_sha256(schema_path),
            "training_view_contract_sha256": file_sha256(train_contract),
            "fit_view_manifest_sha256": file_sha256(fit_man),
            "modeling_split_manifest_sha256": file_sha256(split_path),
            "hgb_model_artifact_sha256": hgb_sha,
            "phase2b2_run_complete_status": b2.get("status"),
        },
        "artifacts": {
            "feature_schema": to_repo_relative(schema_path, project_root=root),
            "training_view_contract": to_repo_relative(
                train_contract, project_root=root
            ),
            "fit_view_manifest": to_repo_relative(fit_man, project_root=root),
            "modeling_split_manifest": to_repo_relative(split_path, project_root=root),
            "hgb_source_artifact": to_repo_relative(hgb_src, project_root=root),
        },
        "scope_limits": [
            "Exactly three families: HGB, AdaBoost, Random Forest.",
            "No ExtraTrees, XGBoost, CatBoost, NN, SVM, KNN, grids, or ensembles.",
            "No TEST access.",
            "Feature 22-vs-27 decision deferred until after family review.",
        ],
    }


def prepare_model_family_bakeoff(
    *,
    project_root: Path | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Write model_family_contract.json and stop (no training)."""
    root = (project_root or PROJECT_ROOT).resolve()
    out = Path(output_dir or DEFAULT_MODEL_FAMILY_ROOT)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    (out / "model_family_complete.json").unlink(missing_ok=True)
    payload = build_model_family_contract(project_root=root)
    contract_path = out / "model_family_contract.json"
    _atomic_json(contract_path, payload)
    return {
        "status": "prepared",
        "strategy_version": MODEL_FAMILY_VERSION,
        "contract_path": to_repo_relative(contract_path, project_root=root),
        "feature_selection_status": payload["feature_selection_status"],
        "next": (
            "Review model_family_contract.json, then run "
            "`iot-pcap-pipeline run-model-family-bakeoff`."
        ),
    }


def load_model_family_contract(
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
            f"model_family_contract.json missing: {p}. "
            "Run prepare-model-family-bakeoff first."
        )
    payload = json.loads(p.read_text(encoding="utf-8"))
    if payload.get("strategy_version") != MODEL_FAMILY_VERSION:
        raise FeatureExtractionError(
            f"unexpected model-family strategy_version: "
            f"{payload.get('strategy_version')!r}"
        )
    if payload.get("feature_selection_status") != "deferred":
        raise FeatureExtractionError(
            "contract must record feature_selection_status=deferred"
        )
    if payload.get("bakeoff_feature_set") != "all_27_v1":
        raise FeatureExtractionError(
            "contract must record bakeoff_feature_set=all_27_v1"
        )
    return payload


def score_bakeoff_tape(
    estimator: Any,
    *,
    project_root: Path,
    split_manifest_path: Path,
    expected_rows: int = EXPECTED_VAL_ROWS,
    progress_file: TextIO | None = None,
) -> BakeoffScoreTape:
    specs = load_validation_specs(split_manifest_path, project_root=project_root)
    validate_validation_inventory(specs, smoke_only=False)

    y_true = np.empty(expected_rows, dtype=np.uint8)
    scores = np.empty(expected_rows, dtype=np.float32)
    group_code = np.empty(expected_rows, dtype=np.uint8)
    pcap_code = np.empty(expected_rows, dtype=np.uint8)
    cursor = 0
    pcap_table: list[dict[str, str]] = []
    pcap_index: dict[str, int] = {}

    for batch in iter_validation_batches(specs, project_root=project_root):
        n = batch.X.shape[0]
        if cursor + n > y_true.shape[0]:
            need = cursor + n - y_true.shape[0]
            y_true = np.concatenate([y_true, np.empty(need, dtype=np.uint8)])
            scores = np.concatenate([scores, np.empty(need, dtype=np.float32)])
            group_code = np.concatenate(
                [group_code, np.empty(need, dtype=np.uint8)]
            )
            pcap_code = np.concatenate(
                [pcap_code, np.empty(need, dtype=np.uint8)]
            )
        batch_scores = attack_score_from_estimator(estimator, batch.X)
        code = _group_code_for_spec(batch.spec)
        pid = batch.spec.pcap_id
        if pid not in pcap_index:
            if len(pcap_table) >= 256:
                raise FeatureExtractionError("pcap_code overflow (>255 PCAPs)")
            pcap_index[pid] = len(pcap_table)
            pcap_table.append(
                {
                    "pcap_id": pid,
                    "modeling_group_key": batch.spec.modeling_group_key,
                    "binary_label": batch.spec.binary_label,
                    "benign_category": batch.spec.benign_category,
                }
            )
        idx = pcap_index[pid]
        y_true[cursor : cursor + n] = batch.y
        scores[cursor : cursor + n] = batch_scores
        group_code[cursor : cursor + n] = code
        pcap_code[cursor : cursor + n] = idx
        cursor += n
        if progress_file is not None and cursor % 500_000 < n:
            progress_file.write(f"  scored {cursor} validation rows\n")
            progress_file.flush()

    if cursor != expected_rows:
        raise FeatureExtractionError(
            f"validation score tape length {cursor} != expected {expected_rows}"
        )
    if len(pcap_table) != EXPECTED_VAL_PCAPS:
        raise FeatureExtractionError(
            f"validation PCAPs scored {len(pcap_table)} != {EXPECTED_VAL_PCAPS}"
        )
    return BakeoffScoreTape(
        y_true=y_true[:cursor],
        scores=scores[:cursor],
        group_code=group_code[:cursor],
        pcap_code=pcap_code[:cursor],
        pcap_table=pcap_table,
    )


def macro_benign_pcap_fpr(tape: BakeoffScoreTape, *, threshold: float) -> float | None:
    fprs: list[float] = []
    for i, meta in enumerate(tape.pcap_table):
        if meta["binary_label"] != "BENIGN":
            continue
        mask = tape.pcap_code == i
        if not np.any(mask):
            continue
        y = tape.y_true[mask]
        s = tape.scores[mask]
        neg = int(np.sum(y == 0))
        if neg == 0:
            continue
        fp = int(np.sum((s >= threshold) & (y == 0)))
        fprs.append(float(fp) / float(neg))
    return macro_mean(fprs)


def metrics_at_threshold_bakeoff(
    tape: BakeoffScoreTape,
    *,
    threshold: float,
    model_id: str,
    point_type: str,
    fpr_target: float | None = None,
) -> dict[str, Any]:
    row = metrics_at_threshold(
        tape.as_validation_tape(),
        threshold=threshold,
        model_id=model_id,
        point_type=point_type,
        fpr_target=fpr_target,
    )
    row["macro_benign_pcap_fpr"] = macro_benign_pcap_fpr(tape, threshold=threshold)
    return row


def eval_tables_at_threshold(
    tape: BakeoffScoreTape,
    *,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, Any]:
    """Build 2B.2-style group/PCAP tables at a fixed reference threshold."""
    y = tape.y_true
    s = tape.scores
    pred = (s >= threshold).astype(np.uint8)
    global_counts = ConfusionCounts()
    global_counts.update(y, pred)

    attack_groups: dict[str, GroupAccumulator] = {
        g: GroupAccumulator(key=g, kind="attack_group", binary_label="ATTACK")
        for g in ATTACK_VAL_GROUPS
    }
    benign_groups: dict[str, GroupAccumulator] = {}
    pcap_acc: dict[str, GroupAccumulator] = {}
    pcap_meta: dict[str, dict[str, str]] = {}

    # Process per PCAP to preserve pcap_id associations without re-streaming.
    for i, meta in enumerate(tape.pcap_table):
        mask = tape.pcap_code == i
        if not np.any(mask):
            continue
        pid = meta["pcap_id"]
        yb = y[mask]
        sb = s[mask]
        pb = pred[mask]
        pcap_acc[pid] = GroupAccumulator(
            key=pid, kind="pcap", binary_label=meta["binary_label"]
        )
        pcap_meta[pid] = meta
        pcap_acc[pid].update(pcap_id=pid, y_true=yb, y_pred=pb, scores=sb)

        if meta["binary_label"] == "ATTACK":
            gkey = meta["modeling_group_key"]
            if gkey in attack_groups:
                attack_groups[gkey].update(
                    pcap_id=pid, y_true=yb, y_pred=pb, scores=sb
                )
        else:
            bkey = benign_group_key(meta["benign_category"], meta["modeling_group_key"])
            if bkey is not None:
                if bkey not in benign_groups:
                    benign_groups[bkey] = GroupAccumulator(
                        key=bkey, kind="benign_group", binary_label="BENIGN"
                    )
                benign_groups[bkey].update(
                    pcap_id=pid, y_true=yb, y_pred=pb, scores=sb
                )

    attack_rows = [attack_groups[g].to_attack_row() for g in ATTACK_VAL_GROUPS]
    benign_rows = [benign_groups[k].to_benign_row() for k in sorted(benign_groups)]
    pcap_rows = [
        pcap_acc[pid].to_pcap_row(
            modeling_group_key=pcap_meta[pid]["modeling_group_key"],
            binary_label=pcap_meta[pid]["binary_label"],
            benign_category=pcap_meta[pid]["benign_category"],
        )
        for pid in sorted(pcap_acc)
    ]
    ranking = global_ranking_metrics(y, s)
    threshold_metrics = metrics_from_confusion(global_counts, threshold=threshold)
    return {
        "n_rows": int(y.shape[0]),
        "n_attack": int((y == 1).sum()),
        "n_benign": int((y == 0).sum()),
        "global": {**ranking, **threshold_metrics},
        "macros": {
            "macro_attack_group_recall": macro_mean(
                [r["recall"] for r in attack_rows]
            ),
            "macro_pcap_attack_recall": macro_mean(
                [
                    r["recall"]
                    for r in pcap_rows
                    if r["binary_label"] == "ATTACK"
                ]
            ),
            "macro_benign_pcap_fpr": macro_mean(
                [r["fpr"] for r in pcap_rows if r["binary_label"] == "BENIGN"]
            ),
        },
        "attack_group_rows": attack_rows,
        "benign_group_rows": benign_rows,
        "pcap_rows": pcap_rows,
        "validation_pcaps_scored": len(pcap_acc),
    }


def _save_score_cache(path: Path, tape: BakeoffScoreTape) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        y_true=tape.y_true,
        scores=tape.scores,
        group_code=tape.group_code,
        pcap_code=tape.pcap_code,
    )
    table_path = path.with_name(path.stem + "_pcap_table.json")
    _atomic_json(table_path, {"pcap_table": tape.pcap_table})


def _load_score_cache(path: Path) -> BakeoffScoreTape:
    cached = np.load(path)
    table_path = path.with_name(path.stem + "_pcap_table.json")
    if not table_path.is_file():
        raise FeatureExtractionError(f"missing pcap table cache: {table_path}")
    table_payload = json.loads(table_path.read_text(encoding="utf-8"))
    return BakeoffScoreTape(
        y_true=cached["y_true"],
        scores=cached["scores"],
        group_code=cached["group_code"],
        pcap_code=cached["pcap_code"],
        pcap_table=list(table_payload["pcap_table"]),
    )


def _write_family_artifacts(
    *,
    model_dir: Path,
    model_id: str,
    display_name: str,
    hyperparameters: dict[str, Any],
    model_path: Path,
    model_sha: str,
    fit_seconds: float,
    score_seconds: float,
    tape: BakeoffScoreTape,
    project_root: Path,
    extra_metadata: dict[str, Any] | None = None,
    reused: bool = False,
    strategy_version: str | None = None,
) -> dict[str, Any]:
    eval_payload = eval_tables_at_threshold(tape, threshold=DECISION_THRESHOLD)
    metadata: dict[str, Any] = {
        "model_id": model_id,
        "display_name": display_name,
        "strategy_version": strategy_version or MODEL_FAMILY_VERSION,
        "hyperparameters": hyperparameters,
        "feature_names": list(V1_FEATURE_NAMES),
        "feature_count": len(V1_FEATURE_NAMES),
        "feature_selection_status": "deferred",
        "bakeoff_feature_set": "all_27_v1",
        "label_mapping": dict(LABEL_MAPPING),
        "decision_threshold": DECISION_THRESHOLD,
        "class_weights": "none",
        "fit_rows": EXPECTED_FIT_ROWS,
        "fit_attack_rows": EXPECTED_FIT_ATTACK,
        "fit_benign_rows": EXPECTED_FIT_BENIGN,
        "fit_duration_seconds": fit_seconds,
        "validation_duration_seconds": score_seconds,
        "validation_rows_scored": EXPECTED_VAL_ROWS,
        "validation_pcaps_scored": EXPECTED_VAL_PCAPS,
        "test_pcaps_read": 0,
        "random_seed": RANDOM_SEED,
        "versions": _package_versions(),
        "model_artifact": to_repo_relative(model_path, project_root=project_root),
        "model_artifact_sha256": model_sha,
        "reused_phase2b2": reused,
        "score_note": (
            "Attack-class predict_proba score is not a calibrated real-world "
            "probability under the resampled FIT view."
        ),
        "threshold_tuning": False,
        "test_access": False,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    metrics = {
        "model_id": model_id,
        "validation_rows": eval_payload["n_rows"],
        "validation_attack_rows": eval_payload["n_attack"],
        "validation_benign_rows": eval_payload["n_benign"],
        "validation_pcaps_scored": eval_payload["validation_pcaps_scored"],
        "global": eval_payload["global"],
        "macros": eval_payload["macros"],
        "decision_threshold": DECISION_THRESHOLD,
        "note": (
            "metrics.json uses threshold 0.5 for continuity with Phase 2B.2; "
            "model-family selection uses comparison_low_fpr.csv."
        ),
    }
    _atomic_json(model_dir / "model_metadata.json", metadata)
    _atomic_json(model_dir / "metrics.json", metrics)
    _atomic_csv(
        model_dir / "attack_group_metrics.csv",
        eval_payload["attack_group_rows"],
        [
            "modeling_group_key",
            "pcap_count",
            "row_count",
            "tp",
            "fn",
            "recall",
            "attack_score_mean",
            "attack_score_p05",
            "attack_score_p50",
            "attack_score_p95",
        ],
    )
    _atomic_csv(
        model_dir / "benign_group_metrics.csv",
        eval_payload["benign_group_rows"],
        [
            "benign_group",
            "pcap_count",
            "row_count",
            "tn",
            "fp",
            "fpr",
            "specificity",
            "attack_score_mean",
            "attack_score_p95",
            "attack_score_p99",
            "max_attack_score",
        ],
    )
    _atomic_csv(
        model_dir / "pcap_metrics.csv",
        eval_payload["pcap_rows"],
        [
            "pcap_id",
            "modeling_group_key",
            "binary_label",
            "benign_category",
            "row_count",
            "tp",
            "fp",
            "tn",
            "fn",
            "recall",
            "fpr",
            "specificity",
            "attack_score_mean",
            "attack_score_p95",
            "attack_score_p99",
            "max_attack_score",
        ],
    )
    return {
        "metadata": metadata,
        "metrics": metrics,
        "roc_auc": eval_payload["global"].get("roc_auc"),
        "pr_auc": eval_payload["global"].get("pr_auc"),
    }


def run_model_family_bakeoff(
    *,
    project_root: Path | None = None,
    output_dir: Path | str | None = None,
    fit_manifest_path: Path | str | None = None,
    split_manifest_path: Path | str | None = None,
    progress_file: TextIO | None = None,
    cache_scores: bool = True,
) -> dict[str, Any]:
    """Train AdaBoost/RF, reuse HGB, score full TRAIN-validation, compare."""
    root = (project_root or PROJECT_ROOT).resolve()
    out = Path(output_dir or DEFAULT_MODEL_FAMILY_ROOT)
    if not out.is_absolute():
        out = root / out

    contract = load_model_family_contract(
        out / "model_family_contract.json", project_root=root
    )
    pinned_hgb = str(
        (contract.get("pins") or {}).get("hgb_model_artifact_sha256") or ""
    )

    require_fit_view_ready(project_root=root, smoke_only=False)
    require_phase2b2_complete(project_root=root)

    fit_man = Path(fit_manifest_path or DEFAULT_FIT_VIEW_MANIFEST_PATH)
    if not fit_man.is_absolute():
        fit_man = root / fit_man
    split_path = Path(split_manifest_path or DEFAULT_SPLIT_MANIFEST_PATH)
    if not split_path.is_absolute():
        split_path = root / split_path

    out.mkdir(parents=True, exist_ok=True)
    (out / "model_family_complete.json").unlink(missing_ok=True)
    models_dir = out / "models"
    pred_dir = out / "predictions"
    models_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    if progress_file is not None:
        progress_file.write("Loading FIT arrays (27 features)...\n")
        progress_file.flush()
    fit = load_fit_arrays(fit_man, project_root=root, smoke_only=False)
    if fit.n_rows != EXPECTED_FIT_ROWS:
        raise FeatureExtractionError(f"FIT rows {fit.n_rows} != {EXPECTED_FIT_ROWS}")
    if fit.n_attack != EXPECTED_FIT_ATTACK or fit.n_benign != EXPECTED_FIT_BENIGN:
        raise FeatureExtractionError(
            f"FIT label counts mismatch: attack={fit.n_attack} "
            f"benign={fit.n_benign}"
        )

    all_fixed: list[dict[str, Any]] = []
    all_low: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    family_summaries: list[dict[str, Any]] = []

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
        reused = bool(spec["reuse_phase2b2"])
        if reused:
            src = root / HGB_SOURCE_JOBLIB
            src_sha = file_sha256(src)
            if pinned_hgb and src_sha != pinned_hgb:
                raise FeatureExtractionError(
                    f"HGB SHA changed since contract freeze: "
                    f"now={src_sha} pinned={pinned_hgb}"
                )
            shutil.copy2(src, model_path)
            model_sha = file_sha256(model_path)
            if model_sha != src_sha:
                raise FeatureExtractionError("HGB copy SHA mismatch")
            estimator = joblib.load(model_path)
            if progress_file is not None:
                progress_file.write(f"  reused 2B.2 HGB sha={model_sha[:12]}...\n")
                progress_file.flush()
        else:
            if progress_file is not None:
                progress_file.write(f"  fitting {model_id}...\n")
                progress_file.flush()
            builder = spec["builder"]
            assert builder is not None
            estimator = builder()
            t0 = time.perf_counter()
            estimator.fit(fit.X, fit.y)
            fit_seconds = time.perf_counter() - t0
            joblib.dump(estimator, model_path)
            model_sha = file_sha256(model_path)

        # Sanity check class order / predict_proba.
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
                f"{model_id}: scored rows {tape.n_rows} != {EXPECTED_VAL_ROWS}"
            )
        if len(tape.pcap_table) != EXPECTED_VAL_PCAPS:
            raise FeatureExtractionError(
                f"{model_id}: scored PCAPs {len(tape.pcap_table)} != "
                f"{EXPECTED_VAL_PCAPS}"
            )

        extra_meta: dict[str, Any] = {}
        if model_id == "adaboost":
            extra_meta["paper_reference"] = dict(ADABOOST_PAPER_REFERENCE)
            extra_meta["implementation"] = dict(ADABOOST_IMPLEMENTATION_NOTE)
            extra_meta["compatibility_note"] = ADABOOST_IMPLEMENTATION_NOTE[
                "compatibility_note"
            ]

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
            extra_metadata=extra_meta or None,
            reused=reused,
        )

        fixed_rows = [
            metrics_at_threshold_bakeoff(
                tape,
                threshold=t,
                model_id=model_id,
                point_type="fixed_threshold",
            )
            for t in FIXED_THRESHOLDS
        ]
        low_rows: list[dict[str, Any]] = []
        for target in LOW_FPR_TARGETS:
            thr = threshold_for_benign_fpr(tape.as_validation_tape(), target)
            low_rows.append(
                metrics_at_threshold_bakeoff(
                    tape,
                    threshold=thr,
                    model_id=model_id,
                    point_type="fpr_target",
                    fpr_target=target,
                )
            )
        all_fixed.extend(fixed_rows)
        all_low.extend(low_rows)
        ranking_rows.append(
            {
                "model_id": model_id,
                "roc_auc": art["roc_auc"],
                "pr_auc": art["pr_auc"],
                "accuracy_at_0_5": art["metrics"]["global"].get("accuracy"),
                "f1_at_0_5": art["metrics"]["global"].get("f1"),
                "benign_fpr_at_0_5": art["metrics"]["global"].get("benign_fpr"),
                "note": "secondary metrics only; do not auto-select winner",
            }
        )
        family_summaries.append(
            {
                "model_id": model_id,
                "model_artifact_sha256": model_sha,
                "fit_seconds": fit_seconds,
                "score_seconds": score_seconds,
                "validation_rows_scored": EXPECTED_VAL_ROWS,
                "validation_pcaps_scored": EXPECTED_VAL_PCAPS,
                "test_pcaps_read": 0,
                "reused_phase2b2": reused,
            }
        )

    _atomic_csv(
        out / "comparison_fixed_thresholds.csv",
        all_fixed,
        list(BAKEOFF_SWEEP_COLUMNS),
    )
    _atomic_csv(
        out / "comparison_low_fpr.csv",
        all_low,
        list(BAKEOFF_SWEEP_COLUMNS),
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
            "note",
        ],
    )

    complete = {
        "status": "passed",
        "strategy_version": MODEL_FAMILY_VERSION,
        "models_evaluated": ["hgb", "adaboost", "random_forest"],
        "feature_selection_status": "deferred",
        "bakeoff_feature_set": "all_27_v1",
        "final_feature_count": "unresolved",
        "feature_count": 27,
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
            "sampling": "never",
            "rows_scored_per_model": EXPECTED_VAL_ROWS,
            "pcaps_scored_per_model": EXPECTED_VAL_PCAPS,
        },
        "test": {"access": False, "pcaps_read": 0},
        "low_fpr_targets": list(LOW_FPR_TARGETS),
        "ranking_criteria": list(RANKING_CRITERIA),
        "auto_declare_winner": False,
        "winner": None,
        "families": family_summaries,
        "artifacts": {
            "model_family_contract": to_repo_relative(
                out / "model_family_contract.json", project_root=root
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
            "Review comparison_low_fpr.csv using the predeclared ranking "
            "criteria. Do not declare a winner in code. Feature 22-vs-27 "
            "remains deferred. Do not touch TEST."
        ),
    }
    _atomic_json(out / "model_family_complete.json", complete)
    return complete


def format_prepare_model_family_summary(payload: dict[str, Any]) -> str:
    return (
        "Phase 2B.4 — model-family contract prepared\n"
        f"status: {payload.get('status')}\n"
        f"strategy_version: {payload.get('strategy_version')}\n"
        f"feature_selection_status: {payload.get('feature_selection_status')}\n"
        f"contract: {payload.get('contract_path')}\n"
        f"next: {payload.get('next')}\n"
    )


def format_model_family_summary(payload: dict[str, Any]) -> str:
    lines = [
        "Phase 2B.4 — model-family bake-off",
        f"status: {payload.get('status')}",
        f"strategy_version: {payload.get('strategy_version')}",
        f"models: {payload.get('models_evaluated')}",
        f"feature_selection_status: {payload.get('feature_selection_status')}",
        f"bakeoff_feature_set: {payload.get('bakeoff_feature_set')}",
        f"final_feature_count: {payload.get('final_feature_count')}",
        f"auto_declare_winner: {payload.get('auto_declare_winner')}",
        f"winner: {payload.get('winner')}",
    ]
    for fam in payload.get("families") or []:
        lines.append(
            f"  {fam['model_id']}: sha={str(fam['model_artifact_sha256'])[:12]}… "
            f"fit_s={fam['fit_seconds']:.1f} score_s={fam['score_seconds']:.1f} "
            f"rows={fam['validation_rows_scored']} "
            f"pcaps={fam['validation_pcaps_scored']} "
            f"test={fam['test_pcaps_read']}"
        )
    arts = payload.get("artifacts") or {}
    if arts.get("comparison_low_fpr"):
        lines.append(f"low_fpr: {arts['comparison_low_fpr']}")
    if payload.get("next"):
        lines.append(f"next: {payload['next']}")
    return "\n".join(lines) + "\n"
