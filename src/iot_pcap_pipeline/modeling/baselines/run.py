"""Orchestrate Phase 2B.2 unweighted baseline training + validation eval."""

from __future__ import annotations

import csv
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import joblib
import numpy as np

from iot_pcap_pipeline.features.schema import V1_FEATURE_NAMES
from iot_pcap_pipeline.modeling.baselines.constants import (
    ATTACK_VAL_GROUPS,
    BASELINE_STRATEGY_VERSION,
    DECISION_THRESHOLD,
    EXPECTED_FIT_ATTACK,
    EXPECTED_FIT_BENIGN,
    EXPECTED_FIT_PCAPS,
    EXPECTED_FIT_ROWS,
    EXPECTED_VAL_PCAPS,
    EXPECTED_VAL_ROWS,
    LABEL_MAPPING,
)
from iot_pcap_pipeline.modeling.baselines.contract import (
    DEFAULT_BASELINES_ROOT,
    build_baseline_contract_payload,
    require_fit_view_ready,
    verify_pinned_hashes,
    write_baseline_contract,
)
from iot_pcap_pipeline.modeling.baselines.data import (
    assert_fit_val_disjoint,
    iter_validation_batches,
    load_fit_arrays,
    load_fit_manifest_rows,
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
    MODEL_SPECS,
    RANDOM_SEED,
    attack_score_from_estimator,
)
from iot_pcap_pipeline.modeling.view import (
    DEFAULT_FIT_VIEW_COMPLETE_PATH,
    DEFAULT_FIT_VIEW_MANIFEST_PATH,
    DEFAULT_SPLIT_MANIFEST_PATH,
    file_sha256,
)
from iot_pcap_pipeline.paths import PROJECT_ROOT, to_repo_relative
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError

SMOKE_MAX_FIT_ROWS = 8_000
SMOKE_MAX_VAL_ROWS = 4_000


@dataclass
class BaselineRunResult:
    passed: bool
    smoke_only: bool
    contract: dict[str, Any]
    comparison_rows: list[dict[str, Any]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    output_dir: Path | None = None
    run_complete_path: Path | None = None
    test_pcaps_read: int = 0


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


def _joblib_sha256(path: Path) -> str:
    return file_sha256(path)


def evaluate_model_on_validation(
    estimator: Any,
    val_specs: list[Any],
    *,
    project_root: Path,
    expected_rows: int,
    max_rows: int | None,
    threshold: float = DECISION_THRESHOLD,
    progress_file: TextIO | None = None,
) -> dict[str, Any]:
    """Stream validation PCAPs; keep only y_true + attack scores globally."""
    y_true = np.empty(expected_rows, dtype=np.uint8)
    scores = np.empty(expected_rows, dtype=np.float32)
    cursor = 0
    global_counts = ConfusionCounts()
    attack_groups: dict[str, GroupAccumulator] = {
        g: GroupAccumulator(key=g, kind="attack_group", binary_label="ATTACK")
        for g in ATTACK_VAL_GROUPS
    }
    benign_groups: dict[str, GroupAccumulator] = {}
    pcap_acc: dict[str, GroupAccumulator] = {}
    pcap_meta: dict[str, dict[str, str]] = {}

    for batch in iter_validation_batches(
        val_specs,
        project_root=project_root,
        max_rows=max_rows,
    ):
        spec = batch.spec
        if cursor + batch.X.shape[0] > y_true.shape[0]:
            need = cursor + batch.X.shape[0] - y_true.shape[0]
            y_true = np.concatenate(
                [y_true, np.empty(need, dtype=np.uint8)]
            )
            scores = np.concatenate(
                [scores, np.empty(need, dtype=np.float32)]
            )

        batch_scores = attack_score_from_estimator(estimator, batch.X)
        batch_pred = (batch_scores >= threshold).astype(np.uint8)
        n = batch.X.shape[0]
        y_true[cursor : cursor + n] = batch.y
        scores[cursor : cursor + n] = batch_scores
        cursor += n
        global_counts.update(batch.y, batch_pred)

        if spec.pcap_id not in pcap_acc:
            pcap_acc[spec.pcap_id] = GroupAccumulator(
                key=spec.pcap_id,
                kind="pcap",
                binary_label=spec.binary_label,
            )
            pcap_meta[spec.pcap_id] = {
                "modeling_group_key": spec.modeling_group_key,
                "binary_label": spec.binary_label,
                "benign_category": spec.benign_category,
            }
        pcap_acc[spec.pcap_id].update(
            pcap_id=spec.pcap_id,
            y_true=batch.y,
            y_pred=batch_pred,
            scores=batch_scores,
        )

        if spec.binary_label == "ATTACK":
            gkey = spec.modeling_group_key
            if gkey in attack_groups:
                attack_groups[gkey].update(
                    pcap_id=spec.pcap_id,
                    y_true=batch.y,
                    y_pred=batch_pred,
                    scores=batch_scores,
                )
        else:
            bkey = benign_group_key(spec.benign_category, spec.modeling_group_key)
            if bkey is not None:
                if bkey not in benign_groups:
                    benign_groups[bkey] = GroupAccumulator(
                        key=bkey, kind="benign_group", binary_label="BENIGN"
                    )
                benign_groups[bkey].update(
                    pcap_id=spec.pcap_id,
                    y_true=batch.y,
                    y_pred=batch_pred,
                    scores=batch_scores,
                )

        if progress_file is not None and cursor % 500_000 < n:
            progress_file.write(f"  validation rows scored: {cursor}\n")
            progress_file.flush()

    y_true = y_true[:cursor]
    scores = scores[:cursor]
    ranking = global_ranking_metrics(y_true, scores)
    threshold_metrics = metrics_from_confusion(global_counts, threshold=threshold)

    attack_rows = [attack_groups[g].to_attack_row() for g in ATTACK_VAL_GROUPS]
    benign_rows = [
        benign_groups[k].to_benign_row()
        for k in sorted(benign_groups)
    ]
    pcap_rows = []
    for pid in sorted(pcap_acc):
        meta = pcap_meta[pid]
        pcap_rows.append(
            pcap_acc[pid].to_pcap_row(
                modeling_group_key=meta["modeling_group_key"],
                binary_label=meta["binary_label"],
                benign_category=meta["benign_category"],
            )
        )

    attack_pcap_recalls = [
        r["recall"]
        for r in pcap_rows
        if r["binary_label"] == "ATTACK"
    ]
    benign_pcap_fprs = [
        r["fpr"] for r in pcap_rows if r["binary_label"] == "BENIGN"
    ]

    return {
        "n_rows": cursor,
        "n_attack": int((y_true == 1).sum()),
        "n_benign": int((y_true == 0).sum()),
        "global": {
            **ranking,
            **threshold_metrics,
        },
        "macros": {
            "macro_attack_group_recall": macro_mean(
                [r["recall"] for r in attack_rows]
            ),
            "macro_pcap_attack_recall": macro_mean(attack_pcap_recalls),
            "macro_benign_pcap_fpr": macro_mean(benign_pcap_fprs),
        },
        "attack_group_rows": attack_rows,
        "benign_group_rows": benign_rows,
        "pcap_rows": pcap_rows,
    }


def _train_one(
    *,
    spec: dict[str, Any],
    fit: Any,
    val_specs: list[Any],
    model_dir: Path,
    project_root: Path,
    smoke_only: bool,
    max_val_rows: int | None,
    expected_val_rows: int,
    progress_file: TextIO | None,
) -> dict[str, Any]:
    model_id = str(spec["model_id"])
    if progress_file is not None:
        progress_file.write(f"Training {model_id}...\n")
        progress_file.flush()

    estimator = spec["builder"]()
    t0 = time.perf_counter()
    convergence_warning: str | None = None
    try:
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(fit.X, fit.y)
            for w in caught:
                if "ConvergenceWarning" in w.category.__name__:
                    convergence_warning = str(w.message)
    except Exception as exc:  # noqa: BLE001 — record training failure cleanly
        raise FeatureExtractionError(
            f"{model_id} training failed: {exc}"
        ) from exc
    fit_seconds = time.perf_counter() - t0

    # Sanity: class order
    _ = attack_score_from_estimator(estimator, fit.X[: min(16, fit.n_rows)])

    models_dir = model_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{model_id}.joblib"
    joblib.dump(estimator, model_path)
    model_sha = _joblib_sha256(model_path)

    if progress_file is not None:
        progress_file.write(f"Evaluating {model_id} on TRAIN-validation...\n")
        progress_file.flush()
    t1 = time.perf_counter()
    eval_payload = evaluate_model_on_validation(
        estimator,
        val_specs,
        project_root=project_root,
        expected_rows=expected_val_rows,
        max_rows=max_val_rows,
        progress_file=progress_file,
    )
    val_seconds = time.perf_counter() - t1

    metadata = {
        "model_id": model_id,
        "display_name": spec["display_name"],
        "strategy_version": BASELINE_STRATEGY_VERSION,
        "hyperparameters": spec["hyperparameters"],
        "feature_names": list(V1_FEATURE_NAMES),
        "feature_count": len(V1_FEATURE_NAMES),
        "label_mapping": dict(LABEL_MAPPING),
        "decision_threshold": DECISION_THRESHOLD,
        "class_weights": "none",
        "fit_rows": fit.n_rows,
        "fit_attack_rows": fit.n_attack,
        "fit_benign_rows": fit.n_benign,
        "fit_duration_seconds": fit_seconds,
        "validation_duration_seconds": val_seconds,
        "random_seed": RANDOM_SEED,
        "versions": _package_versions(),
        "model_artifact": to_repo_relative(model_path, project_root=project_root),
        "model_artifact_sha256": model_sha,
        "convergence_warning": convergence_warning,
        "score_note": (
            "Attack-class predict_proba score is not a calibrated real-world "
            "probability under the resampled FIT view."
        ),
        "smoke_only": smoke_only,
        "early_stopping": False,
        "threshold_tuning": False,
        "test_access": False,
    }
    metrics = {
        "model_id": model_id,
        "smoke_only": smoke_only,
        "validation_rows": eval_payload["n_rows"],
        "validation_attack_rows": eval_payload["n_attack"],
        "validation_benign_rows": eval_payload["n_benign"],
        "global": eval_payload["global"],
        "macros": eval_payload["macros"],
        "decision_threshold": DECISION_THRESHOLD,
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

    g = eval_payload["global"]
    comparison = {
        "model_id": model_id,
        "smoke_only": str(smoke_only).lower(),
        "roc_auc": g.get("roc_auc"),
        "pr_auc": g.get("pr_auc"),
        "precision": g.get("precision"),
        "attack_recall": g.get("attack_recall"),
        "f1": g.get("f1"),
        "specificity": g.get("specificity"),
        "benign_fpr": g.get("benign_fpr"),
        "benign_fp_count": g.get("benign_fp_count"),
        "benign_support": g.get("benign_support"),
        "false_positives_per_10k_benign": g.get("false_positives_per_10k_benign"),
        "balanced_accuracy": g.get("balanced_accuracy"),
        "accuracy": g.get("accuracy"),
        "macro_attack_group_recall": eval_payload["macros"]["macro_attack_group_recall"],
        "macro_pcap_attack_recall": eval_payload["macros"]["macro_pcap_attack_recall"],
        "macro_benign_pcap_fpr": eval_payload["macros"]["macro_benign_pcap_fpr"],
        "fit_duration_seconds": fit_seconds,
        "validation_duration_seconds": val_seconds,
        "convergence_warning": convergence_warning or "",
    }
    return {
        "metadata": metadata,
        "metrics": metrics,
        "comparison": comparison,
        "eval": eval_payload,
    }


def train_baselines(
    *,
    smoke: bool = False,
    project_root: Path | None = None,
    output_dir: Path | str | None = None,
    fit_complete_path: Path | str | None = None,
    fit_manifest_path: Path | str | None = None,
    split_manifest_path: Path | str | None = None,
    progress_file: TextIO | None = None,
) -> BaselineRunResult:
    """Train LR + HGB on FIT view; evaluate on unsampled TRAIN-validation."""
    root = (project_root or PROJECT_ROOT).resolve()
    smoke_only = bool(smoke)
    out = Path(
        output_dir
        or (
            DEFAULT_BASELINES_ROOT.parent / f"{BASELINE_STRATEGY_VERSION}_smoke"
            if smoke_only
            else DEFAULT_BASELINES_ROOT
        )
    )
    if not out.is_absolute():
        out = root / out

    require_fit_view_ready(
        fit_complete_path=fit_complete_path or DEFAULT_FIT_VIEW_COMPLETE_PATH,
        project_root=root,
        smoke_only=smoke_only,
    )

    fit_man = Path(fit_manifest_path or DEFAULT_FIT_VIEW_MANIFEST_PATH)
    if not fit_man.is_absolute():
        fit_man = root / fit_man
    split_path = Path(split_manifest_path or DEFAULT_SPLIT_MANIFEST_PATH)
    if not split_path.is_absolute():
        split_path = root / split_path

    contract_path = out / "baseline_contract.json"
    # Drop stale complete marker until acceptance.
    (out / "run_complete.json").unlink(missing_ok=True)

    write_baseline_contract(
        contract_path,
        project_root=root,
        smoke_only=smoke_only,
        fit_view_manifest_path=fit_man,
        split_manifest_path=split_path,
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    verify_pinned_hashes(contract, project_root=root)

    if progress_file is not None:
        progress_file.write("Loading FIT arrays...\n")
        progress_file.flush()

    max_fit = SMOKE_MAX_FIT_ROWS if smoke_only else None
    fit = load_fit_arrays(
        fit_man,
        project_root=root,
        expected_rows=None if smoke_only else EXPECTED_FIT_ROWS,
        max_rows=max_fit,
        smoke_only=smoke_only,
    )
    val_specs = load_validation_specs(split_path, project_root=root)
    val_inventory = validate_validation_inventory(val_specs, smoke_only=smoke_only)
    assert_fit_val_disjoint(fit.pcap_ids, val_specs)

    # Ensure no validation PCAP is in the fit manifest.
    fit_rows = load_fit_manifest_rows(fit_man)
    fit_ids = {r["pcap_id"] for r in fit_rows}
    val_ids = {s.pcap_id for s in val_specs}
    if fit_ids & val_ids:
        raise FeatureExtractionError(
            f"FIT/VAL pcap_id overlap: {sorted(fit_ids & val_ids)[:5]}"
        )

    max_val = SMOKE_MAX_VAL_ROWS if smoke_only else None
    expected_val = (
        min(SMOKE_MAX_VAL_ROWS, val_inventory["validation_rows"])
        if smoke_only
        else EXPECTED_VAL_ROWS
    )

    comparison_rows: list[dict[str, Any]] = []
    model_results: list[dict[str, Any]] = []
    issues: list[str] = []

    for spec in MODEL_SPECS:
        model_dir = out / str(spec["model_id"])
        try:
            result = _train_one(
                spec=spec,
                fit=fit,
                val_specs=val_specs,
                model_dir=model_dir,
                project_root=root,
                smoke_only=smoke_only,
                max_val_rows=max_val,
                expected_val_rows=expected_val,
                progress_file=progress_file,
            )
            model_results.append(result)
            comparison_rows.append(result["comparison"])
        except FeatureExtractionError as exc:
            issues.append(str(exc))

    _atomic_csv(
        out / "comparison.csv",
        comparison_rows,
        [
            "model_id",
            "smoke_only",
            "roc_auc",
            "pr_auc",
            "precision",
            "attack_recall",
            "f1",
            "specificity",
            "benign_fpr",
            "benign_fp_count",
            "benign_support",
            "false_positives_per_10k_benign",
            "balanced_accuracy",
            "accuracy",
            "macro_attack_group_recall",
            "macro_pcap_attack_recall",
            "macro_benign_pcap_fpr",
            "fit_duration_seconds",
            "validation_duration_seconds",
            "convergence_warning",
        ],
    )

    # Acceptance (full run only enforces frozen totals).
    if not smoke_only:
        if fit.n_rows != EXPECTED_FIT_ROWS:
            issues.append(f"FIT rows {fit.n_rows} != {EXPECTED_FIT_ROWS}")
        if fit.n_attack != EXPECTED_FIT_ATTACK:
            issues.append(f"FIT attack {fit.n_attack} != {EXPECTED_FIT_ATTACK}")
        if fit.n_benign != EXPECTED_FIT_BENIGN:
            issues.append(f"FIT benign {fit.n_benign} != {EXPECTED_FIT_BENIGN}")
        if len(fit.pcap_ids) != EXPECTED_FIT_PCAPS:
            # smoke may truncate pcaps list early; full must load all shards
            if len(load_fit_manifest_rows(fit_man)) != EXPECTED_FIT_PCAPS:
                issues.append("FIT PCAP count mismatch")
        if val_inventory["validation_pcaps"] != EXPECTED_VAL_PCAPS:
            issues.append("validation PCAP count mismatch")
        if val_inventory["validation_rows"] != EXPECTED_VAL_ROWS:
            issues.append("validation row count mismatch")
        for result in model_results:
            if int(result["eval"]["n_rows"]) != EXPECTED_VAL_ROWS:
                issues.append(
                    f"{result['metadata']['model_id']}: validation scored "
                    f"{result['eval']['n_rows']} != {EXPECTED_VAL_ROWS}"
                )
        if len(model_results) != len(MODEL_SPECS):
            issues.append("not all models trained successfully")

    passed = not issues
    run_complete = {
        "status": "passed" if passed else "failed",
        "strategy_version": BASELINE_STRATEGY_VERSION,
        "smoke_only": smoke_only,
        "label_mapping": dict(LABEL_MAPPING),
        "decision_threshold": DECISION_THRESHOLD,
        "class_weights": "none",
        "threshold_tuning": False,
        "test_access": False,
        "test_pcaps_read": 0,
        "fit": {
            "pcaps": len(load_fit_manifest_rows(fit_man)) if not smoke_only else len(fit.pcap_ids),
            "rows": fit.n_rows,
            "attack_rows": fit.n_attack,
            "benign_rows": fit.n_benign,
        },
        "validation": {
            "pcaps": val_inventory["validation_pcaps"],
            "rows_inventory": val_inventory["validation_rows"],
            "rows_scored": (
                model_results[0]["eval"]["n_rows"] if model_results else 0
            ),
            "sampling": "never",
        },
        "models": [r["metadata"]["model_id"] for r in model_results],
        "issues": issues,
        "artifacts": {
            "baseline_contract": to_repo_relative(contract_path, project_root=root),
            "comparison": to_repo_relative(out / "comparison.csv", project_root=root),
            "output_dir": to_repo_relative(out, project_root=root),
        },
        "next": (
            "Review benign FP/FPR, Recon/MQTT/DDoS/DoS holdout recall, and LR vs "
            "HGB gap. Do not pick a winner, tune thresholds, add weights, or "
            "consult TEST."
        ),
    }
    complete_path = out / "run_complete.json"
    if passed:
        _atomic_json(complete_path, run_complete)
    else:
        # Still write a failed marker for debugging, but status=failed.
        _atomic_json(complete_path, run_complete)

    return BaselineRunResult(
        passed=passed,
        smoke_only=smoke_only,
        contract=contract,
        comparison_rows=comparison_rows,
        issues=issues,
        output_dir=out,
        run_complete_path=complete_path,
        test_pcaps_read=0,
    )


def format_baselines_summary(result: BaselineRunResult) -> str:
    lines = [
        "Phase 2B.2 — unweighted binary baselines",
        f"status: {'passed' if result.passed else 'FAILED'}",
        f"smoke_only: {str(result.smoke_only).lower()}",
        f"output_dir: {result.output_dir}",
    ]
    for row in result.comparison_rows:
        lines.append(
            f"{row['model_id']}: roc_auc={row.get('roc_auc')} "
            f"pr_auc={row.get('pr_auc')} "
            f"benign_fp={row.get('benign_fp_count')}/"
            f"{row.get('benign_support')} "
            f"fpr={row.get('benign_fpr')} "
            f"attack_recall={row.get('attack_recall')}"
        )
    if result.issues:
        lines.append("issues:")
        for issue in result.issues:
            lines.append(f"  - {issue}")
    else:
        lines.append(
            "next: inspect metrics; do not tune, weight, ablate, or touch TEST."
        )
    return "\n".join(lines) + "\n"
