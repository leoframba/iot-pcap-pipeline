"""Independent D1 parity: classify_pcap vs explicit reference pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import dpkt
import joblib
import numpy as np
import pytest
from pcap_synth import eth_ip_tcp, write_pcap

from iot_pcap_pipeline.features.extractor import extract_features
from iot_pcap_pipeline.paths import PROJECT_ROOT
from iot_pcap_pipeline.pcap.decode import DLT_EN10MB
from iot_pcap_pipeline.pcap.reader import iter_packets
from iot_pcap_pipeline.serving.aggregate import aggregate_window_scores
from iot_pcap_pipeline.serving.classify import classify_pcap
from iot_pcap_pipeline.serving.contract import (
    EXPECTED_MODEL_SHA256,
    WINDOW_ATTACK_THRESHOLD,
    load_model_input_feature_names,
    sha256_file,
)
from iot_pcap_pipeline.serving.errors import STATUS_OK
from iot_pcap_pipeline.serving.labels import ATTACK_CLASS, BENIGN_CLASS
from iot_pcap_pipeline.serving.model import V1InferenceEngine
from iot_pcap_pipeline.windowing.policy import WINDOW_SIZE
from iot_pcap_pipeline.windowing.stream import iter_windows

# Small FIT PCAPs for optional corpus parity (raw + built feature shards).
_CORPUS_FIT_CASES = (
    {
        "pcap_id": "Singcall_WAN_PHYSICAL-648acf97ea26a9fb",
        "pcap_path": (
            "data/raw/WiFI_and_MQTT/profiling/PCAP/Interactions/Singcall/"
            "Singcall_WAN_PHYSICAL.pcap"
        ),
        "feature_parquet_path": (
            "data/features/v1/train/Singcall_WAN_PHYSICAL-648acf97ea26a9fb.parquet"
        ),
    },
    {
        "pcap_id": "Recon-Ping_Sweep_train-e08ccc2a81c89ea0",
        "pcap_path": (
            "data/raw/WiFI_and_MQTT/attacks/pcap/train/Recon-Ping_Sweep_train.pcap"
        ),
        "feature_parquet_path": (
            "data/features/v1/train/Recon-Ping_Sweep_train-e08ccc2a81c89ea0.parquet"
        ),
    },
)


def _reference_score_windows(
    pcap_path: Path,
    *,
    estimator,
    feature_names_22: list[str],
    attack_class_index: int,
) -> tuple[list[float], object]:
    """Explicit reference path — does not call classify_pcap helpers."""
    rows: list[list[float]] = []
    for window in iter_windows(iter_packets(pcap_path)):
        fv = extract_features(window)
        data = fv.to_feature_dict()
        rows.append([float(data[name]) for name in feature_names_22])

    if not rows:
        scores: list[float] = []
    else:
        X = np.asarray(rows, dtype=np.float32)
        proba = estimator.predict_proba(X)
        scores = [float(v) for v in np.asarray(proba[:, attack_class_index]).tolist()]

    agg = aggregate_window_scores(scores)
    return scores, agg


def _load_reference_estimator():
    model_path = PROJECT_ROOT / "artifacts" / "v1" / "H0_full_fit.joblib"
    assert sha256_file(model_path) == EXPECTED_MODEL_SHA256
    estimator = joblib.load(model_path)
    classes = list(getattr(estimator, "classes_", []))
    assert BENIGN_CLASS in classes and ATTACK_CLASS in classes
    return estimator, int(classes.index(ATTACK_CLASS))


def test_classify_pcap_matches_independent_synthetic_reference(tmp_path: Path) -> None:
    n_windows = 5
    packets = [
        (float(i) * 0.001, eth_ip_tcp(sport=2000 + i, flags=dpkt.tcp.TH_SYN))
        for i in range(WINDOW_SIZE * n_windows)
    ]
    path = write_pcap(tmp_path / "parity.pcap", packets, linktype=DLT_EN10MB)

    feature_names_22 = load_model_input_feature_names()
    estimator, attack_idx = _load_reference_estimator()
    ref_scores, ref_agg = _reference_score_windows(
        path,
        estimator=estimator,
        feature_names_22=feature_names_22,
        attack_class_index=attack_idx,
    )

    engine = V1InferenceEngine.load_default()
    # Independent serving-side score tape (public APIs only; not classify helpers).
    serving_rows: list[list[float]] = []
    for window in iter_windows(iter_packets(path)):
        data = extract_features(window).to_feature_dict()
        serving_rows.append([float(data[name]) for name in engine.feature_names])
    serving_scores = [
        float(s)
        for s in engine.score_matrix(np.asarray(serving_rows, dtype=np.float32)).tolist()
    ]
    np.testing.assert_allclose(serving_scores, ref_scores, rtol=0, atol=1e-7)

    # Force batch boundaries different from default to stress streaming path.
    got = classify_pcap(path, engine=engine, batch_size=2)

    assert got.status == STATUS_OK
    assert ref_agg.status == STATUS_OK
    assert got.window_summary["total_windows"] == len(ref_scores) == n_windows
    assert got.window_summary["attack_windows"] == ref_agg.window_summary.attack_windows
    assert got.window_summary["max_window_attack_score"] == pytest.approx(
        ref_agg.window_summary.max_window_attack_score
    )
    assert got.window_summary["mean_window_attack_score"] == pytest.approx(
        ref_agg.window_summary.mean_window_attack_score
    )
    assert got.pcap_attack_score == pytest.approx(ref_agg.pcap_attack_score)
    assert got.prediction == ref_agg.prediction
    assert sum(1 for s in ref_scores if s >= WINDOW_ATTACK_THRESHOLD) == (
        got.window_summary["attack_windows"]
    )

    # Model response block matches frozen D0 schema (no per-prediction SHA).
    assert set(got.model.keys()) == {
        "model_version",
        "serving_contract_version",
        "score_semantics",
    }
    assert "model_artifact_sha256" not in got.model


@pytest.mark.corpus
@pytest.mark.parametrize("case", _CORPUS_FIT_CASES, ids=lambda c: c["pcap_id"])
def test_classify_pcap_matches_fit_feature_shard_scores(case: dict[str, str]) -> None:
    """Raw-PCAP serving scores match already-built FIT parquet rows (local CIC)."""
    import pyarrow.parquet as pq

    pcap_path = PROJECT_ROOT / case["pcap_path"]
    parquet_path = PROJECT_ROOT / case["feature_parquet_path"]
    if not pcap_path.is_file() or not parquet_path.is_file():
        pytest.skip("local FIT PCAP or feature shard missing")

    # Confirm this PCAP is a FIT row in the frozen split manifest.
    with (PROJECT_ROOT / "data/modeling/v1/modeling_split_manifest.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    match = [r for r in rows if r["pcap_id"] == case["pcap_id"]]
    assert len(match) == 1
    assert match[0]["modeling_split"] == "fit"

    feature_names_22 = load_model_input_feature_names()
    estimator, attack_idx = _load_reference_estimator()

    table = pq.read_table(parquet_path, columns=feature_names_22)
    X = np.column_stack(
        [table.column(name).to_numpy(zero_copy_only=False) for name in feature_names_22]
    ).astype(np.float32, copy=False)
    shard_scores = np.asarray(
        estimator.predict_proba(X)[:, attack_idx], dtype=np.float64
    )

    ref_scores, ref_agg = _reference_score_windows(
        pcap_path,
        estimator=estimator,
        feature_names_22=feature_names_22,
        attack_class_index=attack_idx,
    )
    engine = V1InferenceEngine.load_default()
    got = classify_pcap(pcap_path, engine=engine, batch_size=64)

    assert len(ref_scores) == int(X.shape[0]) == int(match[0]["window_count"])
    np.testing.assert_allclose(ref_scores, shard_scores, rtol=0, atol=1e-6)

    assert got.status == ref_agg.status
    assert got.window_summary["total_windows"] == len(ref_scores)
    assert got.window_summary["attack_windows"] == ref_agg.window_summary.attack_windows
    assert got.window_summary["max_window_attack_score"] == pytest.approx(
        float(np.max(shard_scores))
    )
    assert got.window_summary["mean_window_attack_score"] == pytest.approx(
        float(np.mean(shard_scores))
    )
    assert got.pcap_attack_score == pytest.approx(ref_agg.pcap_attack_score)
    assert got.prediction == ref_agg.prediction


def test_d1_complete_artifact_present() -> None:
    path = PROJECT_ROOT / "data" / "serving" / "v1" / "d1_complete.json"
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["phase"] == "D1"
    assert payload["status"] == "complete"
    assert payload["model_sha256"] == EXPECTED_MODEL_SHA256
    assert payload["feature_count"] == 22
    assert payload["window_size"] == 25
    assert payload["score_batch_size"] == 1024
    assert payload["accepted_linktype"] == 1
    assert payload["parity_status"] == "passed"
    assert payload["serving_contract_version"] == "v1"
