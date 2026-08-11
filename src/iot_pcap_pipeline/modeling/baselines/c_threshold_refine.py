"""Phase 2B.3C: focused threshold refine on C (22-feature unweighted HGB)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from iot_pcap_pipeline.modeling.baselines.ablations import (
    ABLATION_VERSION,
    DEFAULT_ABLATION_ROOT,
)
from iot_pcap_pipeline.modeling.baselines.constants import EXPECTED_VAL_ROWS
from iot_pcap_pipeline.modeling.baselines.model_input import (
    DROPPED_TEMPORAL_FEATURES,
    V1_MODEL_INPUT_FEATURES,
    V1_MODEL_INPUT_VERSION,
    write_v1_model_input_contract,
)
from iot_pcap_pipeline.modeling.baselines.threshold_sweep import (
    SWEEP_ROW_COLUMNS,
    ValidationScoreTape,
    metrics_at_threshold,
    threshold_for_benign_fpr,
)
from iot_pcap_pipeline.paths import (
    DEFAULT_MODELING_DIR,
    PROJECT_ROOT,
    to_repo_relative,
)
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError

REFINE_VERSION = "phase2b3c_v1"
DEFAULT_REFINE_ROOT = DEFAULT_MODELING_DIR / "v1" / "baselines" / REFINE_VERSION

VARIANT_ID = "C_22_unweighted"

# Focused low-FPR band for provisional threshold selection (not a broad grid).
FOCUSED_FPR_TARGETS: tuple[float, ...] = (
    0.005,  # ~0.5%
    0.0025,  # ~0.25%
    0.001,  # ~0.1%
    0.0005,  # ~0.05%
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def load_c_score_tape(
    *,
    ablation_dir: Path,
    project_root: Path,
) -> ValidationScoreTape:
    score_cache = ablation_dir / "predictions" / f"{VARIANT_ID}_val_scores.npz"
    if not score_cache.is_file():
        raise FeatureExtractionError(
            f"missing C validation score cache: {score_cache}; "
            "run `iot-pcap-pipeline run-hgb-ablations` first"
        )
    model_path = ablation_dir / "models" / f"{VARIANT_ID}.joblib"
    if not model_path.is_file():
        raise FeatureExtractionError(
            f"missing C model artifact: {model_path}; "
            "run `iot-pcap-pipeline run-hgb-ablations` first"
        )
    complete = ablation_dir / "ablation_complete.json"
    if not complete.is_file():
        raise FeatureExtractionError(
            f"missing ablation complete marker: {complete}"
        )
    cached = np.load(score_cache)
    tape = ValidationScoreTape(
        y_true=cached["y_true"],
        scores=cached["scores"],
        group_code=cached["group_code"],
    )
    if tape.n_rows != EXPECTED_VAL_ROWS:
        raise FeatureExtractionError(
            f"C cache rows {tape.n_rows} != {EXPECTED_VAL_ROWS}"
        )
    _ = project_root
    return tape


def run_c_threshold_refine(
    *,
    project_root: Path | None = None,
    output_dir: Path | str | None = None,
    ablation_dir: Path | str | None = None,
    progress_file: TextIO | None = None,
) -> dict[str, Any]:
    """Reuse frozen C scores; tabulate focused benign-FPR operating points."""
    root = (project_root or PROJECT_ROOT).resolve()
    abl = Path(ablation_dir or DEFAULT_ABLATION_ROOT)
    if not abl.is_absolute():
        abl = root / abl
    out = Path(output_dir or DEFAULT_REFINE_ROOT)
    if not out.is_absolute():
        out = root / out

    if progress_file is not None:
        progress_file.write(
            f"Loading C ({VARIANT_ID}) scores from {abl / 'predictions'}...\n"
        )
        progress_file.flush()

    tape = load_c_score_tape(ablation_dir=abl, project_root=root)
    contract_path = write_v1_model_input_contract(project_root=root)

    out.mkdir(parents=True, exist_ok=True)
    (out / "refine_complete.json").unlink(missing_ok=True)

    rows: list[dict[str, Any]] = []
    for target in FOCUSED_FPR_TARGETS:
        thr = threshold_for_benign_fpr(tape, target)
        rows.append(
            metrics_at_threshold(
                tape,
                threshold=thr,
                model_id=VARIANT_ID,
                point_type="fpr_target",
                fpr_target=target,
            )
        )
        if progress_file is not None:
            r = rows[-1]
            progress_file.write(
                f"  FPR≤{target:.4%} → thr={thr:.6f} "
                f"benign_fpr={r['benign_fpr']:.6%} "
                f"recon={r['recon_os_scan_recall']:.4%} "
                f"mqtt={r['mqtt_publish_recall']:.4%}\n"
            )
            progress_file.flush()

    contract_payload = {
        "strategy_version": REFINE_VERSION,
        "parent_ablation": ABLATION_VERSION,
        "variant_id": VARIANT_ID,
        "model_input_version": V1_MODEL_INPUT_VERSION,
        "model_input_features": list(V1_MODEL_INPUT_FEATURES),
        "excluded_temporal_features": list(DROPPED_TEMPORAL_FEATURES),
        "class_weight": None,
        "focused_benign_fpr_targets": list(FOCUSED_FPR_TARGETS),
        "validation_sampling": "never",
        "test_access": False,
        "score_source": to_repo_relative(
            abl / "predictions" / f"{VARIANT_ID}_val_scores.npz",
            project_root=root,
        ),
        "model_source": to_repo_relative(
            abl / "models" / f"{VARIANT_ID}.joblib",
            project_root=root,
        ),
        "model_input_contract": to_repo_relative(contract_path, project_root=root),
        "notes": [
            "No retraining. Threshold not frozen by this run.",
            "Pipeline still extracts all 27 V1 features; model input is 22.",
        ],
    }
    _atomic_json(out / "refine_contract.json", contract_payload)

    csv_path = out / "focused_fpr_targets.csv"
    _atomic_csv(csv_path, rows, list(SWEEP_ROW_COLUMNS))

    complete = {
        "status": "passed",
        "strategy_version": REFINE_VERSION,
        "variant_id": VARIANT_ID,
        "model_input_version": V1_MODEL_INPUT_VERSION,
        "focused_benign_fpr_targets": list(FOCUSED_FPR_TARGETS),
        "operating_points": [
            {
                "fpr_target": float(r["fpr_target"]),
                "threshold": float(r["threshold"]),
                "benign_fpr": r["benign_fpr"],
                "owltron_interaction_fpr": r["owltron_interaction_fpr"],
                "profiling_idle_fpr": r["profiling_idle_fpr"],
                "ddos_tcp_recall": r["ddos_tcp_recall"],
                "dos_tcp_recall": r["dos_tcp_recall"],
                "mqtt_publish_recall": r["mqtt_publish_recall"],
                "recon_os_scan_recall": r["recon_os_scan_recall"],
                "macro_attack_group_recall": r["macro_attack_group_recall"],
                "min_attack_group_recall": r["min_attack_group_recall"],
            }
            for r in rows
        ],
        "artifacts": {
            "refine_contract": to_repo_relative(
                out / "refine_contract.json", project_root=root
            ),
            "focused_fpr_targets": to_repo_relative(csv_path, project_root=root),
            "model_input_contract": to_repo_relative(contract_path, project_root=root),
            "output_dir": to_repo_relative(out, project_root=root),
        },
        "next": (
            "Review focused_fpr_targets.csv. Choose a provisional threshold for C "
            "from the ~0.5% / 0.25% / 0.1% / 0.05% band; do not freeze TEST yet."
        ),
        "test_access": False,
    }
    _atomic_json(out / "refine_complete.json", complete)
    return complete


def format_c_threshold_refine_summary(payload: dict[str, Any]) -> str:
    lines = [
        "Phase 2B.3C — focused C (22-feature HGB) threshold refine",
        f"status: {payload.get('status')}",
        f"strategy_version: {payload.get('strategy_version')}",
        f"model_input_version: {payload.get('model_input_version')}",
        f"variant: {payload.get('variant_id')}",
        "operating points:",
    ]
    for pt in payload.get("operating_points") or []:
        lines.append(
            f"  FPR≤{float(pt['fpr_target']):.4%} thr={float(pt['threshold']):.6f} "
            f"benign={float(pt['benign_fpr']):.4%} "
            f"recon={float(pt['recon_os_scan_recall']):.2%} "
            f"mqtt={float(pt['mqtt_publish_recall']):.2%} "
            f"owltron={float(pt['owltron_interaction_fpr']):.4%}"
        )
    arts = payload.get("artifacts") or {}
    if arts.get("focused_fpr_targets"):
        lines.append(f"table: {arts['focused_fpr_targets']}")
    if arts.get("model_input_contract"):
        lines.append(f"model_input: {arts['model_input_contract']}")
    if payload.get("next"):
        lines.append(f"next: {payload['next']}")
    return "\n".join(lines) + "\n"
