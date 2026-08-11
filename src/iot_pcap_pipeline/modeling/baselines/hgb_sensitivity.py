"""Phase 2C.1 — FIT-only HGB sensitivity (group-aware 3-fold CV).

Main TRAIN-validation is forbidden during search. After FIT-CV selection is
written to disk, exactly one baseline-vs-winner comparison is allowed.
TEST stays sealed. No adaptive second round.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from iot_pcap_pipeline.features.parquet import feature_schema_sha256
from iot_pcap_pipeline.features.schema import DEFAULT_FEATURE_SCHEMA_PATH, V1_FEATURE_NAMES
from iot_pcap_pipeline.modeling.baselines.ablations import DEFAULT_ABLATION_ROOT
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
    load_fit_arrays,
    load_fit_manifest_rows,
)
from iot_pcap_pipeline.modeling.baselines.model_family import (
    metrics_at_threshold_bakeoff,
    score_bakeoff_tape,
)
from iot_pcap_pipeline.modeling.baselines.model_input import (
    DROPPED_TEMPORAL_FEATURES,
    V1_MODEL_INPUT_CONTRACT_PATH,
    V1_MODEL_INPUT_FEATURES,
    V1_MODEL_INPUT_VERSION,
)
from iot_pcap_pipeline.modeling.baselines.models import RANDOM_SEED, attack_score_from_estimator
from iot_pcap_pipeline.modeling.baselines.threshold_sweep import (
    ValidationScoreTape,
    threshold_for_benign_fpr_with_reachability,
)
from iot_pcap_pipeline.modeling.baselines.v1_candidate_freeze import (
    DEFAULT_V1_CANDIDATE_FREEZE_PATH,
)
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

SENSITIVITY_VERSION = "phase2c1_v1"
DEFAULT_SENSITIVITY_ROOT = (
    DEFAULT_MODELING_DIR / "v1" / "hgb_sensitivity" / SENSITIVITY_VERSION
)

PRIMARY_FPR = 0.0025
SECONDARY_FPR = 0.001
N_FOLDS = 3
BASE_SEED = 42

# Families that participate in min-family ranking (Spoofing excluded from dominance).
SUPPORTED_ATTACK_FAMILIES: tuple[str, ...] = ("DDoS", "DoS", "MQTT", "Recon")

FEATURES_22 = list(V1_MODEL_INPUT_FEATURES)
assert len(FEATURES_22) == 22

H0_BASE: dict[str, Any] = {
    "learning_rate": 0.10,
    "max_iter": 200,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "max_features": 1.0,
    "early_stopping": False,
    "random_state": BASE_SEED,
    "class_weight": None,
}

# Exactly 12 predeclared candidates. Do not extend after seeing results.
SENSITIVITY_CONFIGS: tuple[dict[str, Any], ...] = (
    {"config_id": "H0", "label": "baseline", "overrides": {}},
    {"config_id": "H1", "label": "smaller_trees", "overrides": {"max_leaf_nodes": 15}},
    {"config_id": "H2", "label": "larger_trees", "overrides": {"max_leaf_nodes": 63}},
    {"config_id": "H3", "label": "larger_leaves", "overrides": {"min_samples_leaf": 50}},
    {
        "config_id": "H4",
        "label": "strongly_larger_leaves",
        "overrides": {"min_samples_leaf": 100},
    },
    {"config_id": "H5", "label": "no_l2", "overrides": {"l2_regularization": 0.0}},
    {"config_id": "H6", "label": "more_l2", "overrides": {"l2_regularization": 5.0}},
    {"config_id": "H7", "label": "strong_l2", "overrides": {"l2_regularization": 10.0}},
    {
        "config_id": "H8",
        "label": "slower_boosting",
        "overrides": {"learning_rate": 0.05, "max_iter": 400},
    },
    {
        "config_id": "H9",
        "label": "faster_boosting",
        "overrides": {"learning_rate": 0.15, "max_iter": 150},
    },
    {
        "config_id": "H10",
        "label": "feature_subsampling",
        "overrides": {"max_features": 0.8},
    },
    {
        "config_id": "H11",
        "label": "regularized_combination",
        "overrides": {
            "max_leaf_nodes": 31,
            "min_samples_leaf": 50,
            "l2_regularization": 5.0,
            "max_features": 0.8,
        },
    },
)
assert len(SENSITIVITY_CONFIGS) == 12

MATERIAL_RECALL_GAIN = 0.01  # 1 percentage point
MQTT_MATERIAL_DROP = 0.01


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


def resolve_hgb_params(overrides: dict[str, Any]) -> dict[str, Any]:
    params = dict(H0_BASE)
    params.update(overrides)
    params["early_stopping"] = False
    params["random_state"] = BASE_SEED
    params["class_weight"] = None
    return params


def config_params_list() -> list[dict[str, Any]]:
    out = []
    for spec in SENSITIVITY_CONFIGS:
        params = resolve_hgb_params(dict(spec["overrides"]))
        out.append(
            {
                "config_id": spec["config_id"],
                "label": spec["label"],
                "overrides": dict(spec["overrides"]),
                "params": params,
            }
        )
    return out


def _group_hash(group_key: str, *, seed: int = BASE_SEED) -> int:
    digest = hashlib.sha256(
        f"{SENSITIVITY_VERSION}|{seed}|{group_key}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)


@dataclass(frozen=True)
class FitGroupMeta:
    modeling_group_key: str
    attack_family: str
    binary_label: str
    row_count: int
    pcap_ids: tuple[str, ...]


def load_fit_group_metas(fit_manifest_path: Path) -> list[FitGroupMeta]:
    rows = load_fit_manifest_rows(fit_manifest_path)
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["modeling_group_key"])
        if key not in buckets:
            buckets[key] = {
                "attack_family": str(row.get("attack_family") or "BENIGN"),
                "binary_label": str(row["binary_label"]),
                "row_count": 0,
                "pcap_ids": [],
            }
        buckets[key]["row_count"] += int(row["output_row_count"])
        buckets[key]["pcap_ids"].append(str(row["pcap_id"]))
        if not buckets[key]["attack_family"] and row.get("attack_family"):
            buckets[key]["attack_family"] = str(row["attack_family"])
    metas = [
        FitGroupMeta(
            modeling_group_key=k,
            attack_family=v["attack_family"] if v["binary_label"] == "ATTACK" else "BENIGN",
            binary_label=v["binary_label"],
            row_count=int(v["row_count"]),
            pcap_ids=tuple(sorted(v["pcap_ids"])),
        )
        for k, v in buckets.items()
    ]
    metas.sort(key=lambda m: m.modeling_group_key)
    return metas


def assign_fit_cv_folds(
    metas: list[FitGroupMeta],
    *,
    n_folds: int = N_FOLDS,
    seed: int = BASE_SEED,
) -> dict[str, int]:
    """Assign each modeling_group_key to exactly one fold (validation partition)."""
    if n_folds != 3:
        raise FeatureExtractionError("Phase 2C.1 requires exactly 3 folds")
    assignment: dict[str, int] = {}
    fold_rows = [0, 0, 0]

    # Attack families with multiple lineages: distribute whole groups across folds.
    by_family: dict[str, list[FitGroupMeta]] = defaultdict(list)
    benign: list[FitGroupMeta] = []
    spoofing: list[FitGroupMeta] = []
    for meta in metas:
        if meta.binary_label == "BENIGN":
            benign.append(meta)
        elif meta.attack_family == "Spoofing":
            spoofing.append(meta)
        else:
            by_family[meta.attack_family].append(meta)

    for family in SUPPORTED_ATTACK_FAMILIES:
        groups = sorted(
            by_family.get(family, []),
            key=lambda m: (_group_hash(m.modeling_group_key, seed=seed), m.modeling_group_key),
        )
        if not groups:
            continue
        if len(groups) >= n_folds:
            # First n_folds groups: one per fold (ensures each fold holds out a lineage).
            for i, meta in enumerate(groups[:n_folds]):
                assignment[meta.modeling_group_key] = i
                fold_rows[i] += meta.row_count
            # Remainder: greedy to lowest row fold (deterministic order).
            for meta in groups[n_folds:]:
                fold = int(np.argmin(fold_rows))
                assignment[meta.modeling_group_key] = fold
                fold_rows[fold] += meta.row_count
        else:
            for meta in groups:
                fold = _group_hash(meta.modeling_group_key, seed=seed) % n_folds
                assignment[meta.modeling_group_key] = fold
                fold_rows[fold] += meta.row_count

    # Spoofing: single lineage — assign deterministically; excluded from ranking dominance.
    for meta in sorted(
        spoofing,
        key=lambda m: (_group_hash(m.modeling_group_key, seed=seed), m.modeling_group_key),
    ):
        fold = _group_hash(meta.modeling_group_key, seed=seed) % n_folds
        assignment[meta.modeling_group_key] = fold
        fold_rows[fold] += meta.row_count

    # Benign: greedy balance by row count (atomic groups).
    for meta in sorted(
        benign,
        key=lambda m: (_group_hash(m.modeling_group_key, seed=seed), m.modeling_group_key),
    ):
        fold = int(np.argmin(fold_rows))
        assignment[meta.modeling_group_key] = fold
        fold_rows[fold] += meta.row_count

    if len(assignment) != len(metas):
        raise FeatureExtractionError("fold assignment missing groups")
    return assignment


def expand_row_group_labels(
    fit_manifest_path: Path,
    *,
    expected_rows: int = EXPECTED_FIT_ROWS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (group_key object array, attack_family object array, fold int array).

    Aligned to load_fit_arrays row order (manifest sorted by pcap_id).
    """
    rows = load_fit_manifest_rows(fit_manifest_path)
    metas = load_fit_group_metas(fit_manifest_path)
    fold_of = assign_fit_cv_folds(metas)
    group_keys = np.empty(expected_rows, dtype=object)
    families = np.empty(expected_rows, dtype=object)
    folds = np.empty(expected_rows, dtype=np.int8)
    cursor = 0
    for row in rows:
        n = int(row["output_row_count"])
        gkey = str(row["modeling_group_key"])
        fam = str(row.get("attack_family") or "BENIGN")
        if str(row["binary_label"]) == "BENIGN":
            fam = "BENIGN"
        fold = fold_of[gkey]
        group_keys[cursor : cursor + n] = gkey
        families[cursor : cursor + n] = fam
        folds[cursor : cursor + n] = fold
        cursor += n
    if cursor != expected_rows:
        raise FeatureExtractionError(
            f"expanded FIT rows {cursor} != {expected_rows}"
        )
    return group_keys, families, folds


def write_fit_cv_artifacts(
    *,
    out_dir: Path,
    fit_manifest_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    metas = load_fit_group_metas(fit_manifest_path)
    fold_of = assign_fit_cv_folds(metas)
    rows = load_fit_manifest_rows(fit_manifest_path)

    manifest_rows: list[dict[str, Any]] = []
    fold_row_totals = [0, 0, 0]
    fold_groups: list[list[str]] = [[], [], []]
    for row in rows:
        gkey = str(row["modeling_group_key"])
        fold = fold_of[gkey]
        n = int(row["output_row_count"])
        fold_row_totals[fold] += n
        manifest_rows.append(
            {
                "pcap_id": row["pcap_id"],
                "modeling_group_key": gkey,
                "attack_family": row.get("attack_family") or "BENIGN",
                "binary_label": row["binary_label"],
                "output_row_count": n,
                "fold_id": fold,
                "role_in_fold": "validation_when_held_out",
            }
        )
    for gkey, fold in sorted(fold_of.items()):
        fold_groups[fold].append(gkey)

    # Overlap check: groups unique to one fold.
    seen: dict[str, int] = {}
    for fold, groups in enumerate(fold_groups):
        for g in groups:
            if g in seen and seen[g] != fold:
                raise FeatureExtractionError(f"group overlap for {g}")
            seen[g] = fold

    if sum(fold_row_totals) != EXPECTED_FIT_ROWS:
        raise FeatureExtractionError(
            f"fold row totals {sum(fold_row_totals)} != {EXPECTED_FIT_ROWS}"
        )

    spoof_folds = [
        fold_of[m.modeling_group_key]
        for m in metas
        if m.attack_family == "Spoofing"
    ]

    summary = {
        "strategy_version": SENSITIVITY_VERSION,
        "n_folds": N_FOLDS,
        "base_seed": BASE_SEED,
        "total_validation_row_assignments": EXPECTED_FIT_ROWS,
        "fold_row_totals": {
            f"fold_{i}": fold_row_totals[i] for i in range(N_FOLDS)
        },
        "fold_group_counts": {
            f"fold_{i}": len(fold_groups[i]) for i in range(N_FOLDS)
        },
        "fold_groups": {f"fold_{i}": fold_groups[i] for i in range(N_FOLDS)},
        "group_overlap": 0,
        "test_pcaps": 0,
        "main_train_validation_rows": 0,
        "spoofing_limitation": {
            "note": (
                "Spoofing has only one FIT lineage (ARP_Spoofing); it cannot "
                "provide three-fold lineage validation. Spoofing is excluded "
                "from primary min-family ranking to avoid dominating selection."
            ),
            "assigned_folds": spoof_folds,
        },
        "supported_attack_families_for_ranking": list(SUPPORTED_ATTACK_FAMILIES),
    }
    _atomic_csv(
        out_dir / "fit_cv_fold_manifest.csv",
        manifest_rows,
        [
            "pcap_id",
            "modeling_group_key",
            "attack_family",
            "binary_label",
            "output_row_count",
            "fold_id",
            "role_in_fold",
        ],
    )
    _atomic_json(out_dir / "fit_cv_summary.json", summary)
    return summary


def build_sensitivity_contract(
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = (project_root or PROJECT_ROOT).resolve()
    require_fit_view_ready(project_root=root, smoke_only=False)
    freeze_path = root / DEFAULT_V1_CANDIDATE_FREEZE_PATH
    if not freeze_path.is_file():
        raise FeatureExtractionError(
            f"missing V1 candidate freeze: {freeze_path}. "
            "Run freeze-v1-candidate first."
        )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen":
        raise FeatureExtractionError("V1 candidate freeze not frozen")

    fit_man = root / DEFAULT_FIT_VIEW_MANIFEST_PATH
    train_contract = root / DEFAULT_TRAINING_VIEW_CONTRACT_PATH
    split_path = root / DEFAULT_SPLIT_MANIFEST_PATH
    schema_path = root / DEFAULT_FEATURE_SCHEMA_PATH
    model_input_path = root / V1_MODEL_INPUT_CONTRACT_PATH
    configs = config_params_list()
    config_blob = json.dumps(configs, sort_keys=True, separators=(",", ":"))
    config_sha = hashlib.sha256(config_blob.encode("utf-8")).hexdigest()

    return {
        "strategy_version": SENSITIVITY_VERSION,
        "status": "frozen",
        "model_family": "HistGradientBoostingClassifier",
        "feature_count": 22,
        "model_input_version": V1_MODEL_INPUT_VERSION,
        "feature_names": list(FEATURES_22),
        "excluded_temporal_features": list(DROPPED_TEMPORAL_FEATURES),
        "fit_rows": EXPECTED_FIT_ROWS,
        "fit_pcaps": EXPECTED_FIT_PCAPS,
        "fit_attack_rows": EXPECTED_FIT_ATTACK,
        "fit_benign_rows": EXPECTED_FIT_BENIGN,
        "base_seed": BASE_SEED,
        "selection_data": "FIT_only",
        "main_validation_access_during_search": False,
        "test_access": False,
        "primary_operating_fpr": PRIMARY_FPR,
        "secondary_operating_fpr": SECONDARY_FPR,
        "hyperparameter_search": "controlled_predeclared_sensitivity",
        "adaptive_second_round": False,
        "n_folds": N_FOLDS,
        "n_configs": len(SENSITIVITY_CONFIGS),
        "configs": configs,
        "fixed_for_all_candidates": {
            "early_stopping": False,
            "random_state": BASE_SEED,
            "class_weight": None,
        },
        "material_improvement_bar": {
            "min_family_or_recon_gain": MATERIAL_RECALL_GAIN,
            "mqtt_max_drop": MQTT_MATERIAL_DROP,
            "note": (
                "Replace H0 only for ~1pp+ min-family/Recon gain or clear "
                "stability improvement without material MQTT degradation."
            ),
        },
        "pins": {
            "feature_schema_sha256": feature_schema_sha256(schema_path),
            "training_view_contract_sha256": file_sha256(train_contract),
            "fit_view_manifest_sha256": file_sha256(fit_man),
            "modeling_split_manifest_sha256": file_sha256(split_path),
            "model_input_contract_sha256": file_sha256(model_input_path),
            "configuration_list_sha256": config_sha,
            "v1_candidate_freeze_sha256": file_sha256(freeze_path),
        },
        "artifacts": {
            "fit_view_manifest": to_repo_relative(fit_man, project_root=root),
            "model_input_contract": to_repo_relative(
                model_input_path, project_root=root
            ),
            "v1_candidate_freeze": to_repo_relative(freeze_path, project_root=root),
            "feature_schema": to_repo_relative(schema_path, project_root=root),
        },
        "scope_limits": [
            "Exactly 12 predeclared HGB configs; no Cartesian grid.",
            "No adaptive second round.",
            "Main TRAIN-validation forbidden during search.",
            "TEST forbidden.",
            "Stop hyperparameter exploration permanently after this phase.",
        ],
    }


def prepare_hgb_sensitivity(
    *,
    project_root: Path | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Freeze sensitivity_contract.json + FIT CV fold artifacts (no fitting)."""
    root = (project_root or PROJECT_ROOT).resolve()
    out = Path(output_dir or DEFAULT_SENSITIVITY_ROOT)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    (out / "sensitivity_complete.json").unlink(missing_ok=True)
    (out / "selected_candidate.json").unlink(missing_ok=True)

    contract = build_sensitivity_contract(project_root=root)
    _atomic_json(out / "sensitivity_contract.json", contract)
    fit_man = root / DEFAULT_FIT_VIEW_MANIFEST_PATH
    cv_summary = write_fit_cv_artifacts(
        out_dir=out, fit_manifest_path=fit_man, project_root=root
    )
    return {
        "status": "prepared",
        "strategy_version": SENSITIVITY_VERSION,
        "contract_path": to_repo_relative(
            out / "sensitivity_contract.json", project_root=root
        ),
        "fit_cv_summary": cv_summary,
        "n_configs": len(SENSITIVITY_CONFIGS),
        "next": (
            "Review sensitivity_contract.json and fit_cv_summary.json, then run "
            "`iot-pcap-pipeline run-hgb-sensitivity`."
        ),
    }


def load_sensitivity_contract(
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
            f"sensitivity_contract.json missing: {p}. "
            "Run prepare-hgb-sensitivity first."
        )
    payload = json.loads(p.read_text(encoding="utf-8"))
    if payload.get("strategy_version") != SENSITIVITY_VERSION:
        raise FeatureExtractionError("unexpected sensitivity strategy_version")
    if payload.get("main_validation_access_during_search") is not False:
        raise FeatureExtractionError("main_validation_access_during_search must be false")
    if payload.get("test_access") is not False:
        raise FeatureExtractionError("test_access must be false")
    if payload.get("adaptive_second_round") is not False:
        raise FeatureExtractionError("adaptive_second_round must be false")
    if int(payload.get("n_configs") or 0) != 12:
        raise FeatureExtractionError("n_configs must be 12")
    if int(payload.get("feature_count") or 0) != 22:
        raise FeatureExtractionError("feature_count must be 22")
    return payload


def _select_x22(X: np.ndarray) -> np.ndarray:
    idxs = [list(V1_FEATURE_NAMES).index(name) for name in FEATURES_22]
    return X[:, idxs]


def _family_metrics_at_threshold(
    *,
    y_true: np.ndarray,
    scores: np.ndarray,
    families: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    pred = scores >= threshold
    benign = y_true == 0
    fp = int(np.sum(pred & benign))
    benign_support = int(np.sum(benign))
    benign_fpr = (fp / benign_support) if benign_support else None

    family_recalls: dict[str, float | None] = {}
    for fam in list(SUPPORTED_ATTACK_FAMILIES) + ["Spoofing"]:
        mask = (families == fam) & (y_true == 1)
        support = int(np.sum(mask))
        if support == 0:
            family_recalls[fam] = None
            continue
        tp = int(np.sum(pred & mask))
        family_recalls[fam] = float(tp) / float(support)

    supported = [
        float(family_recalls[f])
        for f in SUPPORTED_ATTACK_FAMILIES
        if family_recalls[f] is not None
    ]
    return {
        "threshold": float(threshold),
        "benign_fp": fp,
        "benign_support": benign_support,
        "benign_fpr": benign_fpr,
        "ddos_recall": family_recalls["DDoS"],
        "dos_recall": family_recalls["DoS"],
        "mqtt_recall": family_recalls["MQTT"],
        "recon_recall": family_recalls["Recon"],
        "spoof_recall": family_recalls["Spoofing"],
        "macro_attack_family_recall": (
            float(sum(supported) / len(supported)) if supported else None
        ),
        "min_supported_family_recall": (
            float(min(supported)) if supported else None
        ),
        "supported_families_present": [
            f for f in SUPPORTED_ATTACK_FAMILIES if family_recalls[f] is not None
        ],
    }


def evaluate_fold_at_fpr(
    *,
    y_true: np.ndarray,
    scores: np.ndarray,
    families: np.ndarray,
    fpr_target: float,
) -> dict[str, Any]:
    tape = ValidationScoreTape(
        y_true=y_true.astype(np.uint8, copy=False),
        scores=scores.astype(np.float32, copy=False),
        group_code=np.zeros(y_true.shape[0], dtype=np.uint8),
    )
    thr, reached = threshold_for_benign_fpr_with_reachability(tape, fpr_target)
    metrics = _family_metrics_at_threshold(
        y_true=y_true, scores=scores, families=families, threshold=thr
    )
    metrics["fpr_target"] = float(fpr_target)
    metrics["target_reached"] = bool(reached)
    # If unreachable, still report best-effort metrics at thr (may violate target).
    if metrics["benign_fpr"] is not None and metrics["benign_fpr"] > fpr_target + 1e-15:
        metrics["target_reached"] = False
    return metrics


def _mean_std(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0)), float(arr.max() - arr.min())


def select_candidate_from_cv(
    summary_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply predeclared ranking; high bar vs H0."""
    by_id = {r["config_id"]: r for r in summary_rows}
    if "H0" not in by_id:
        raise FeatureExtractionError("H0 missing from CV summary")
    h0 = by_id["H0"]

    def sort_key(row: dict[str, Any]) -> tuple:
        return (
            -(row["mean_min_supported_family_recall"] or -1.0),
            -(row["mean_recon_recall"] or -1.0),
            -(row["mean_mqtt_recall"] or -1.0),
            -(row["mean_macro_attack_family_recall"] or -1.0),
            row["std_min_supported_family_recall"] or 1e9,
            row["std_recon_recall"] or 1e9,
            row["mean_benign_fpr"] or 1.0,
        )

    ranked = sorted(summary_rows, key=sort_key)
    best = ranked[0]

    def materially_better(challenger: dict[str, Any], baseline: dict[str, Any]) -> bool:
        # Same-or-better FPR compliance vs H0 (not absolute all-folds, which
        # discrete scores can block for every candidate including the baseline).
        c_reach = int(challenger.get("n_folds_primary_reachable") or 0)
        b_reach = int(baseline.get("n_folds_primary_reachable") or 0)
        if c_reach < b_reach:
            return False
        c_fpr = challenger.get("mean_benign_fpr")
        b_fpr = baseline.get("mean_benign_fpr")
        if c_fpr is not None and b_fpr is not None and c_fpr > b_fpr + 1e-4:
            return False
        c_min = challenger["mean_min_supported_family_recall"]
        b_min = baseline["mean_min_supported_family_recall"]
        c_recon = challenger["mean_recon_recall"]
        b_recon = baseline["mean_recon_recall"]
        c_mqtt = challenger["mean_mqtt_recall"]
        b_mqtt = baseline["mean_mqtt_recall"]
        if None in (c_min, b_min, c_recon, b_recon, c_mqtt, b_mqtt):
            return False
        mqtt_ok = (b_mqtt - c_mqtt) <= MQTT_MATERIAL_DROP + 1e-12
        if not mqtt_ok:
            return False
        gain_min = c_min - b_min
        gain_recon = c_recon - b_recon
        stability = (
            (challenger["std_min_supported_family_recall"] or 0)
            + (challenger["std_recon_recall"] or 0)
            + (challenger["std_mqtt_recall"] or 0)
        )
        base_stability = (
            (baseline["std_min_supported_family_recall"] or 0)
            + (baseline["std_recon_recall"] or 0)
            + (baseline["std_mqtt_recall"] or 0)
        )
        clear_stability = (
            stability + 1e-6 < 0.7 * base_stability and base_stability > 0.005
        )
        recall_win = (
            gain_min >= MATERIAL_RECALL_GAIN - 1e-12
            or gain_recon >= MATERIAL_RECALL_GAIN - 1e-12
        )
        return bool(recall_win or clear_stability)

    selected = h0
    reason = "No challenger cleared the material-improvement bar; keep H0 baseline."
    if best["config_id"] != "H0" and materially_better(best, h0):
        selected = best
        reason = (
            f"{best['config_id']} materially improves on H0 under the "
            "predeclared FIT-CV ranking and improvement bar."
        )
    elif best["config_id"] != "H0":
        reason = (
            f"{best['config_id']} ranked highest but did not clear the material "
            "improvement bar vs H0; keep H0."
        )

    return {
        "strategy_version": SENSITIVITY_VERSION,
        "selection_data": "FIT_only",
        "main_validation_used_for_selection": False,
        "selected_config_id": selected["config_id"],
        "selected_label": selected["label"],
        "selected_params": selected["params"],
        "baseline_config_id": "H0",
        "selection_reason": reason,
        "ranking_top3": [
            {
                "config_id": r["config_id"],
                "mean_min_supported_family_recall": r[
                    "mean_min_supported_family_recall"
                ],
                "mean_recon_recall": r["mean_recon_recall"],
                "mean_mqtt_recall": r["mean_mqtt_recall"],
            }
            for r in ranked[:3]
        ],
        "material_improvement_bar": {
            "min_family_or_recon_gain": MATERIAL_RECALL_GAIN,
            "mqtt_max_drop": MQTT_MATERIAL_DROP,
        },
    }


def run_hgb_sensitivity(
    *,
    project_root: Path | None = None,
    output_dir: Path | str | None = None,
    fit_manifest_path: Path | str | None = None,
    split_manifest_path: Path | str | None = None,
    progress_file: TextIO | None = None,
) -> dict[str, Any]:
    """FIT-CV over 12 configs, write selected_candidate.json, then one VAL compare."""
    root = (project_root or PROJECT_ROOT).resolve()
    out = Path(output_dir or DEFAULT_SENSITIVITY_ROOT)
    if not out.is_absolute():
        out = root / out

    contract = load_sensitivity_contract(
        out / "sensitivity_contract.json", project_root=root
    )
    require_fit_view_ready(project_root=root, smoke_only=False)

    fit_man = Path(fit_manifest_path or DEFAULT_FIT_VIEW_MANIFEST_PATH)
    if not fit_man.is_absolute():
        fit_man = root / fit_man
    split_path = Path(split_manifest_path or DEFAULT_SPLIT_MANIFEST_PATH)
    if not split_path.is_absolute():
        split_path = root / split_path

    out.mkdir(parents=True, exist_ok=True)
    (out / "sensitivity_complete.json").unlink(missing_ok=True)
    models_dir = out / "models"
    pred_dir = out / "predictions"
    models_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Ensure CV artifacts exist / refresh.
    write_fit_cv_artifacts(out_dir=out, fit_manifest_path=fit_man, project_root=root)

    if progress_file is not None:
        progress_file.write("Loading FIT arrays (27→22 features)...\n")
        progress_file.flush()
    fit = load_fit_arrays(fit_man, project_root=root, smoke_only=False)
    if fit.n_rows != EXPECTED_FIT_ROWS:
        raise FeatureExtractionError(f"FIT rows {fit.n_rows} != {EXPECTED_FIT_ROWS}")
    X22 = _select_x22(fit.X)
    y = fit.y
    _group_keys, families, folds = expand_row_group_labels(fit_man)

    cv_rows: list[dict[str, Any]] = []
    configs = config_params_list()

    for spec in configs:
        cid = spec["config_id"]
        params = dict(spec["params"])
        if progress_file is not None:
            progress_file.write(f"\n=== {cid} ({spec['label']}) ===\n")
            progress_file.flush()

        for fold_id in range(N_FOLDS):
            val_mask = folds == fold_id
            train_mask = ~val_mask
            if not np.any(val_mask) or not np.any(train_mask):
                raise FeatureExtractionError(f"{cid} fold {fold_id}: empty split")
            # Leakage check: no group in both.
            train_groups = set(_group_keys[train_mask].tolist())
            val_groups = set(_group_keys[val_mask].tolist())
            overlap = train_groups & val_groups
            if overlap:
                raise FeatureExtractionError(
                    f"group leakage fold {fold_id}: {sorted(overlap)[:5]}"
                )

            if progress_file is not None:
                progress_file.write(
                    f"  fold {fold_id}: train={int(train_mask.sum())} "
                    f"val={int(val_mask.sum())}\n"
                )
                progress_file.flush()

            est = HistGradientBoostingClassifier(**params)
            t0 = time.perf_counter()
            est.fit(X22[train_mask], y[train_mask])
            fit_s = time.perf_counter() - t0
            scores = attack_score_from_estimator(est, X22[val_mask])

            for fpr_name, fpr_target in (
                ("primary", PRIMARY_FPR),
                ("secondary", SECONDARY_FPR),
            ):
                m = evaluate_fold_at_fpr(
                    y_true=y[val_mask],
                    scores=scores,
                    families=families[val_mask],
                    fpr_target=fpr_target,
                )
                cv_rows.append(
                    {
                        "config_id": cid,
                        "label": spec["label"],
                        "fold_id": fold_id,
                        "operating_point": fpr_name,
                        "fpr_target": fpr_target,
                        "target_reached": m["target_reached"],
                        "threshold": m["threshold"],
                        "benign_fp": m["benign_fp"],
                        "benign_support": m["benign_support"],
                        "benign_fpr": m["benign_fpr"],
                        "ddos_recall": m["ddos_recall"],
                        "dos_recall": m["dos_recall"],
                        "mqtt_recall": m["mqtt_recall"],
                        "recon_recall": m["recon_recall"],
                        "spoof_recall": m["spoof_recall"],
                        "macro_attack_family_recall": m["macro_attack_family_recall"],
                        "min_supported_family_recall": m[
                            "min_supported_family_recall"
                        ],
                        "supported_families_present": "|".join(
                            m["supported_families_present"]
                        ),
                        "fit_seconds": fit_s,
                        "train_rows": int(train_mask.sum()),
                        "val_rows": int(val_mask.sum()),
                    }
                )

    _atomic_csv(
        out / "cv_results.csv",
        cv_rows,
        [
            "config_id",
            "label",
            "fold_id",
            "operating_point",
            "fpr_target",
            "target_reached",
            "threshold",
            "benign_fp",
            "benign_support",
            "benign_fpr",
            "ddos_recall",
            "dos_recall",
            "mqtt_recall",
            "recon_recall",
            "spoof_recall",
            "macro_attack_family_recall",
            "min_supported_family_recall",
            "supported_families_present",
            "fit_seconds",
            "train_rows",
            "val_rows",
        ],
    )

    # Aggregate primary operating point across folds.
    summary_rows: list[dict[str, Any]] = []
    for spec in configs:
        cid = spec["config_id"]
        primary = [
            r
            for r in cv_rows
            if r["config_id"] == cid and r["operating_point"] == "primary"
        ]
        if len(primary) != N_FOLDS:
            raise FeatureExtractionError(f"{cid}: expected {N_FOLDS} primary rows")

        def _col(name: str) -> list[float]:
            return [float(r[name]) for r in primary if r[name] is not None and r[name] != ""]

        mean_min, std_min, range_min = _mean_std(_col("min_supported_family_recall"))
        mean_recon, std_recon, range_recon = _mean_std(_col("recon_recall"))
        mean_mqtt, std_mqtt, range_mqtt = _mean_std(_col("mqtt_recall"))
        mean_macro, std_macro, range_macro = _mean_std(_col("macro_attack_family_recall"))
        mean_fpr, _, _ = _mean_std(_col("benign_fpr"))
        summary_rows.append(
            {
                "config_id": cid,
                "label": spec["label"],
                "params": spec["params"],
                "mean_min_supported_family_recall": mean_min,
                "std_min_supported_family_recall": std_min,
                "range_min_supported_family_recall": range_min,
                "mean_recon_recall": mean_recon,
                "std_recon_recall": std_recon,
                "range_recon_recall": range_recon,
                "mean_mqtt_recall": mean_mqtt,
                "std_mqtt_recall": std_mqtt,
                "range_mqtt_recall": range_mqtt,
                "mean_macro_attack_family_recall": mean_macro,
                "std_macro_attack_family_recall": std_macro,
                "range_macro_attack_family_recall": range_macro,
                "mean_benign_fpr": mean_fpr,
                "n_folds_primary_reachable": sum(
                    1 for r in primary if bool(r["target_reached"])
                ),
                "all_folds_primary_reachable": all(
                    bool(r["target_reached"]) for r in primary
                ),
            }
        )

    summary_csv_rows = []
    for r in summary_rows:
        row = {k: v for k, v in r.items() if k != "params"}
        row["params_json"] = json.dumps(r["params"], sort_keys=True)
        summary_csv_rows.append(row)
    _atomic_csv(
        out / "cv_candidate_summary.csv",
        summary_csv_rows,
        [
            "config_id",
            "label",
            "mean_min_supported_family_recall",
            "std_min_supported_family_recall",
            "range_min_supported_family_recall",
            "mean_recon_recall",
            "std_recon_recall",
            "range_recon_recall",
            "mean_mqtt_recall",
            "std_mqtt_recall",
            "range_mqtt_recall",
            "mean_macro_attack_family_recall",
            "std_macro_attack_family_recall",
            "range_macro_attack_family_recall",
            "mean_benign_fpr",
            "n_folds_primary_reachable",
            "all_folds_primary_reachable",
            "params_json",
        ],
    )

    selected = select_candidate_from_cv(summary_rows)
    # AUDIT: write selection BEFORE main TRAIN-validation is scored.
    _atomic_json(out / "selected_candidate.json", selected)
    if progress_file is not None:
        progress_file.write(
            f"\nFIT-CV selection written: {selected['selected_config_id']} "
            f"— {selected['selection_reason']}\n"
        )
        progress_file.write(
            "Unlocking main TRAIN-validation for single baseline-vs-winner compare...\n"
        )
        progress_file.flush()

    # --- Post-selection: full FIT train of selected + baseline compare on VAL ---
    selected_params = dict(selected["selected_params"])
    baseline_params = resolve_hgb_params({})

    def _train_full(params: dict[str, Any], model_id: str) -> Any:
        path = models_dir / f"{model_id}.joblib"
        if progress_file is not None:
            progress_file.write(f"  fitting full-FIT {model_id}...\n")
            progress_file.flush()
        est = HistGradientBoostingClassifier(**params)
        t0 = time.perf_counter()
        est.fit(X22, y)
        elapsed = time.perf_counter() - t0
        joblib.dump(est, path)
        return est, file_sha256(path), elapsed

    # Baseline H0 on full FIT (or reuse C_22 if params match exactly).
    c22_path = root / DEFAULT_ABLATION_ROOT / "models" / "C_22_unweighted.joblib"
    baseline_est: Any
    baseline_sha: str
    baseline_fit_s: float
    # C_22 used HGB_PARAMS without explicit min_samples_leaf/max_features;
    # sklearn defaults min_samples_leaf=20, max_features=1.0 — equivalent to H0.
    if c22_path.is_file() and selected["selected_config_id"] == "H0":
        # Still train fresh H0 for a clean artifact under this phase, and score both.
        pass
    baseline_est, baseline_sha, baseline_fit_s = _train_full(baseline_params, "H0_full_fit")
    if selected["selected_config_id"] == "H0":
        winner_est, winner_sha, winner_fit_s = baseline_est, baseline_sha, baseline_fit_s
        # Copy alias
        joblib.dump(winner_est, models_dir / "selected_full_fit.joblib")
    else:
        winner_est, winner_sha, winner_fit_s = _train_full(
            selected_params, "selected_full_fit"
        )

    if progress_file is not None:
        progress_file.write("  scoring TRAIN-validation (baseline H0)...\n")
        progress_file.flush()
    base_tape = score_bakeoff_tape(
        baseline_est,
        project_root=root,
        split_manifest_path=split_path,
        feature_names=FEATURES_22,
        progress_file=progress_file,
    )
    if selected["selected_config_id"] == "H0":
        win_tape = base_tape
    else:
        if progress_file is not None:
            progress_file.write("  scoring TRAIN-validation (selected)...\n")
            progress_file.flush()
        win_tape = score_bakeoff_tape(
            winner_est,
            project_root=root,
            split_manifest_path=split_path,
            feature_names=FEATURES_22,
            progress_file=progress_file,
        )

    if base_tape.n_rows != EXPECTED_VAL_ROWS or win_tape.n_rows != EXPECTED_VAL_ROWS:
        raise FeatureExtractionError("validation row count mismatch")
    if (
        len(base_tape.pcap_table) != EXPECTED_VAL_PCAPS
        or len(win_tape.pcap_table) != EXPECTED_VAL_PCAPS
    ):
        raise FeatureExtractionError("validation PCAP count mismatch")

    val_rows: list[dict[str, Any]] = []
    for model_id, tape in (("H0_baseline", base_tape), ("selected", win_tape)):
        for fpr_name, fpr_target in (
            ("primary", PRIMARY_FPR),
            ("secondary", SECONDARY_FPR),
            ("fpr_0.5pct", 0.005),
            ("fpr_0.1pct", 0.001),
        ):
            thr, reached = threshold_for_benign_fpr_with_reachability(
                tape.as_validation_tape(), fpr_target
            )
            row = metrics_at_threshold_bakeoff(
                tape,
                threshold=thr,
                model_id=model_id,
                point_type="fpr_target",
                fpr_target=fpr_target,
            )
            val_rows.append(
                {
                    "model_id": model_id,
                    "operating_point": fpr_name,
                    "fpr_target": fpr_target,
                    "target_reached": reached,
                    "threshold": thr,
                    "benign_fpr": row["benign_fpr"],
                    "benign_fp": row["benign_fp"],
                    "ddos_tcp_recall": row["ddos_tcp_recall"],
                    "dos_tcp_recall": row["dos_tcp_recall"],
                    "mqtt_publish_recall": row["mqtt_publish_recall"],
                    "recon_os_scan_recall": row["recon_os_scan_recall"],
                    "macro_attack_group_recall": row["macro_attack_group_recall"],
                    "min_attack_group_recall": row["min_attack_group_recall"],
                    "owltron_interaction_fpr": row["owltron_interaction_fpr"],
                }
            )

    _atomic_csv(
        out / "final_validation_comparison.csv",
        val_rows,
        [
            "model_id",
            "operating_point",
            "fpr_target",
            "target_reached",
            "threshold",
            "benign_fpr",
            "benign_fp",
            "ddos_tcp_recall",
            "dos_tcp_recall",
            "mqtt_publish_recall",
            "recon_os_scan_recall",
            "macro_attack_group_recall",
            "min_attack_group_recall",
            "owltron_interaction_fpr",
        ],
    )

    # Final freeze decision from VAL confirmation (no further tuning).
    def _val_primary(model_id: str) -> dict[str, Any]:
        for r in val_rows:
            if r["model_id"] == model_id and r["operating_point"] == "primary":
                return r
        raise FeatureExtractionError(f"missing primary VAL row for {model_id}")

    base_val = _val_primary("H0_baseline")
    win_val = _val_primary("selected")
    freeze_config_id = "H0"
    freeze_reason = "FIT-CV kept H0; freeze original HGB-22 hyperparameters."
    if selected["selected_config_id"] != "H0":
        # Confirm material improvement on main VAL min/recon without MQTT hit.
        b_min = float(base_val["min_attack_group_recall"])
        w_min = float(win_val["min_attack_group_recall"])
        b_recon = float(base_val["recon_os_scan_recall"])
        w_recon = float(win_val["recon_os_scan_recall"])
        b_mqtt = float(base_val["mqtt_publish_recall"])
        w_mqtt = float(win_val["mqtt_publish_recall"])
        mqtt_ok = (b_mqtt - w_mqtt) <= MQTT_MATERIAL_DROP + 1e-12
        confirmed = mqtt_ok and (
            (w_min - b_min) >= MATERIAL_RECALL_GAIN - 1e-12
            or (w_recon - b_recon) >= MATERIAL_RECALL_GAIN - 1e-12
        )
        if confirmed:
            freeze_config_id = selected["selected_config_id"]
            freeze_reason = (
                f"FIT-CV selected {freeze_config_id} and main TRAIN-validation "
                "confirmed a material improvement; freeze selected hyperparameters."
            )
        else:
            freeze_reason = (
                f"FIT-CV selected {selected['selected_config_id']} but main "
                "TRAIN-validation did not confirm a material gain; freeze H0."
            )

    frozen_params = (
        resolve_hgb_params({})
        if freeze_config_id == "H0"
        else dict(selected["selected_params"])
    )

    complete = {
        "status": "passed",
        "strategy_version": SENSITIVITY_VERSION,
        "feature_count": 22,
        "fit_rows": EXPECTED_FIT_ROWS,
        "fit_pcaps": EXPECTED_FIT_PCAPS,
        "n_folds": N_FOLDS,
        "n_configs": 12,
        "adaptive_second_round": False,
        "early_stopping": False,
        "class_weight": None,
        "selection_data": "FIT_only",
        "main_validation_access_during_search": False,
        "main_validation_comparison": {
            "rows_scored": EXPECTED_VAL_ROWS,
            "pcaps_scored": EXPECTED_VAL_PCAPS,
            "models_compared": ["H0_baseline", "selected"],
        },
        "test": {"access": False, "pcaps_read": 0},
        "selected_candidate_path": to_repo_relative(
            out / "selected_candidate.json", project_root=root
        ),
        "fit_cv_selected_config_id": selected["selected_config_id"],
        "frozen_config_id": freeze_config_id,
        "frozen_params": frozen_params,
        "freeze_reason": freeze_reason,
        "baseline_full_fit_sha256": baseline_sha,
        "selected_full_fit_sha256": winner_sha,
        "baseline_fit_seconds": baseline_fit_s,
        "selected_fit_seconds": winner_fit_s,
        "hyperparameter_exploration_status": "closed",
        "artifacts": {
            "sensitivity_contract": to_repo_relative(
                out / "sensitivity_contract.json", project_root=root
            ),
            "fit_cv_fold_manifest": to_repo_relative(
                out / "fit_cv_fold_manifest.csv", project_root=root
            ),
            "fit_cv_summary": to_repo_relative(
                out / "fit_cv_summary.json", project_root=root
            ),
            "cv_results": to_repo_relative(out / "cv_results.csv", project_root=root),
            "cv_candidate_summary": to_repo_relative(
                out / "cv_candidate_summary.csv", project_root=root
            ),
            "selected_candidate": to_repo_relative(
                out / "selected_candidate.json", project_root=root
            ),
            "final_validation_comparison": to_repo_relative(
                out / "final_validation_comparison.csv", project_root=root
            ),
            "output_dir": to_repo_relative(out, project_root=root),
        },
        "contract_pins": contract.get("pins"),
        "next": (
            "Hyperparameter exploration is closed. Record frozen_params into the "
            "V1 candidate package; proceed to threshold freeze. Do not open another "
            "sensitivity round. TEST remains sealed."
        ),
    }
    _assert_complete_gate(complete)
    _atomic_json(out / "sensitivity_complete.json", complete)
    return complete


def _assert_complete_gate(payload: dict[str, Any]) -> None:
    issues: list[str] = []
    if int(payload.get("feature_count") or -1) != 22:
        issues.append("feature_count != 22")
    if int(payload.get("fit_rows") or -1) != EXPECTED_FIT_ROWS:
        issues.append("wrong FIT rows")
    if int(payload.get("n_folds") or -1) != 3:
        issues.append("n_folds != 3")
    if int(payload.get("n_configs") or -1) != 12:
        issues.append("n_configs != 12")
    if payload.get("adaptive_second_round") is not False:
        issues.append("adaptive tuning")
    if payload.get("early_stopping") is not False:
        issues.append("early_stopping")
    if payload.get("class_weight") is not None:
        issues.append("class_weight")
    if payload.get("main_validation_access_during_search") is not False:
        issues.append("main VAL used during search")
    val = payload.get("main_validation_comparison") or {}
    if int(val.get("rows_scored") or -1) != EXPECTED_VAL_ROWS:
        issues.append("VAL rows")
    if int(val.get("pcaps_scored") or -1) != EXPECTED_VAL_PCAPS:
        issues.append("VAL pcaps")
    if (payload.get("test") or {}).get("access") is not False:
        issues.append("TEST access")
    if int((payload.get("test") or {}).get("pcaps_read", -1)) != 0:
        issues.append("TEST pcaps")
    if not payload.get("frozen_params"):
        issues.append("missing frozen_params")
    if not (payload.get("artifacts") or {}).get("selected_candidate"):
        issues.append("missing selected_candidate artifact")
    if issues:
        raise FeatureExtractionError(
            "sensitivity_complete gate failed: " + "; ".join(issues)
        )


def format_prepare_hgb_sensitivity_summary(payload: dict[str, Any]) -> str:
    return (
        "Phase 2C.1 — HGB sensitivity contract prepared\n"
        f"status: {payload.get('status')}\n"
        f"strategy_version: {payload.get('strategy_version')}\n"
        f"n_configs: {payload.get('n_configs')}\n"
        f"contract: {payload.get('contract_path')}\n"
        f"next: {payload.get('next')}\n"
    )


def format_hgb_sensitivity_summary(payload: dict[str, Any]) -> str:
    lines = [
        "Phase 2C.1 — FIT-only HGB sensitivity",
        f"status: {payload.get('status')}",
        f"strategy_version: {payload.get('strategy_version')}",
        f"fit_cv_selected: {payload.get('fit_cv_selected_config_id')}",
        f"frozen_config: {payload.get('frozen_config_id')}",
        f"freeze_reason: {payload.get('freeze_reason')}",
        f"hyperparameter_exploration_status: "
        f"{payload.get('hyperparameter_exploration_status')}",
        f"test_access: {(payload.get('test') or {}).get('access')}",
    ]
    arts = payload.get("artifacts") or {}
    if arts.get("selected_candidate"):
        lines.append(f"selected_before_val: {arts['selected_candidate']}")
    if arts.get("final_validation_comparison"):
        lines.append(f"val_compare: {arts['final_validation_comparison']}")
    if payload.get("next"):
        lines.append(f"next: {payload['next']}")
    return "\n".join(lines) + "\n"
