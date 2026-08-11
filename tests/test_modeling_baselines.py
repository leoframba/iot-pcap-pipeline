"""Phase 2B.2 baseline unit tests (synthetic; no full corpus train)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from iot_pcap_pipeline.features.parquet import feature_parquet_arrow_schema
from iot_pcap_pipeline.features.schema import V1_FEATURE_NAMES, write_feature_schema
from iot_pcap_pipeline.modeling.baselines.constants import (
    FORBIDDEN_MODEL_COLUMNS,
    LABEL_MAPPING,
)
from iot_pcap_pipeline.modeling.baselines.contract import (
    build_baseline_contract_payload,
    verify_pinned_hashes,
)
from iot_pcap_pipeline.modeling.baselines.data import (
    assert_feature_columns,
    encode_labels,
    load_validation_specs,
    reject_test_path,
)
from iot_pcap_pipeline.modeling.baselines.metrics import (
    ConfusionCounts,
    GroupAccumulator,
    macro_mean,
    metrics_from_confusion,
)
from iot_pcap_pipeline.modeling.baselines.models import (
    RANDOM_SEED,
    attack_score_from_estimator,
    build_hist_gradient_boosting,
    build_logistic_regression,
)
from iot_pcap_pipeline.modeling.baselines.run import train_baselines
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError


def test_label_mapping_frozen() -> None:
    assert LABEL_MAPPING == {"BENIGN": 0, "ATTACK": 1}
    y = encode_labels(["BENIGN", "ATTACK", "BENIGN"])
    assert list(y) == [0, 1, 0]
    with pytest.raises(FeatureExtractionError):
        encode_labels(["malware"])


def test_feature_columns_exact_and_no_metadata() -> None:
    assert_feature_columns(list(V1_FEATURE_NAMES))
    with pytest.raises(FeatureExtractionError, match="order mismatch"):
        assert_feature_columns(list(V1_FEATURE_NAMES)[::-1])
    assert "attack_family" in FORBIDDEN_MODEL_COLUMNS
    with pytest.raises(FeatureExtractionError):
        assert_feature_columns(list(V1_FEATURE_NAMES) + ["attack_family"])


def test_test_paths_rejected() -> None:
    with pytest.raises(FeatureExtractionError, match="TEST path"):
        reject_test_path("data/features/v1/test/foo.parquet")
    reject_test_path("data/features/v1/train/foo.parquet")


def test_confusion_metrics_math() -> None:
    counts = ConfusionCounts()
    y_true = np.array([1, 1, 0, 0, 1, 0], dtype=np.uint8)
    y_pred = np.array([1, 0, 0, 1, 1, 0], dtype=np.uint8)
    counts.update(y_true, y_pred)
    assert counts.as_dict() == {"tp": 2, "fn": 1, "tn": 2, "fp": 1}
    m = metrics_from_confusion(counts)
    assert m["attack_recall"] == pytest.approx(2 / 3)
    assert m["benign_fpr"] == pytest.approx(1 / 3)
    assert m["benign_fp_count"] == 1
    assert m["precision"] == pytest.approx(2 / 3)


def test_zero_positive_zero_negative_subgroup() -> None:
    empty_pos = ConfusionCounts(tp=0, fn=0, tn=5, fp=1)
    m = metrics_from_confusion(empty_pos)
    assert m["attack_recall"] is None
    assert m["benign_fpr"] == pytest.approx(1 / 6)

    empty_neg = ConfusionCounts(tp=3, fn=1, tn=0, fp=0)
    m2 = metrics_from_confusion(empty_neg)
    assert m2["benign_fpr"] is None
    assert m2["attack_recall"] == pytest.approx(0.75)
    assert macro_mean([None, 0.5, 1.0]) == pytest.approx(0.75)


def test_group_accumulator_recall_and_fpr() -> None:
    g = GroupAccumulator(key="Recon|OS_Scan", kind="attack_group", binary_label="ATTACK")
    y_true = np.array([1, 1, 1, 1], dtype=np.uint8)
    y_pred = np.array([1, 1, 0, 1], dtype=np.uint8)
    scores = np.array([0.9, 0.8, 0.2, 0.7], dtype=np.float32)
    g.update(pcap_id="p1", y_true=y_true, y_pred=y_pred, scores=scores)
    row = g.to_attack_row()
    assert row["tp"] == 3 and row["fn"] == 1
    assert row["recall"] == pytest.approx(0.75)
    assert row["pcap_count"] == 1


def test_attack_score_class_ordering() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 27)).astype(np.float32)
    y = np.array([0] * 100 + [1] * 100, dtype=np.uint8)
    X[y == 1, 0] += 2.0
    pipe = build_logistic_regression()
    pipe.fit(X, y)
    scores = attack_score_from_estimator(pipe, X)
    assert scores.shape == (200,)
    assert scores.dtype == np.float32
    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_hgb_deterministic_seed42() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 27)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.uint8)
    a = build_hist_gradient_boosting()
    b = build_hist_gradient_boosting()
    assert a.get_params()["random_state"] == RANDOM_SEED
    assert a.get_params()["early_stopping"] is False
    a.fit(X, y)
    b.fit(X, y)
    sa = attack_score_from_estimator(a, X)
    sb = attack_score_from_estimator(b, X)
    np.testing.assert_allclose(sa, sb, rtol=0, atol=0)


def test_contract_hash_mismatch_hard_failure(tmp_path: Path) -> None:
    schema = tmp_path / "feature_schema.json"
    write_feature_schema(schema)
    split = tmp_path / "modeling_split_manifest.csv"
    split.write_text("pcap_path\na\n", encoding="utf-8")
    fit_man = tmp_path / "fit_view_manifest.csv"
    fit_man.write_text("pcap_id\nx\n", encoding="utf-8")
    train_contract = tmp_path / "training_view_contract.json"
    train_contract.write_text("{}\n", encoding="utf-8")

    payload = build_baseline_contract_payload(
        project_root=tmp_path,
        fit_view_manifest_path=fit_man,
        training_view_contract_path=train_contract,
        split_manifest_path=split,
        feature_schema_path=schema,
        smoke_only=True,
    )
    split.write_text("pcap_path\na\nb\n", encoding="utf-8")
    with pytest.raises(FeatureExtractionError, match="mismatch"):
        verify_pinned_hashes(payload, project_root=tmp_path)


def _write_feature_parquet(path: Path, *, n: int, label: str, pcap_id: str) -> None:
    schema = feature_parquet_arrow_schema()
    rows = []
    for i in range(n):
        row = {
            "pcap_id": pcap_id,
            "binary_label": label,
            "segment_index": 0,
            "window_index": i,
            "packet_index_start": i * 25,
            "packet_index_end": i * 25 + 24,
        }
        for name in V1_FEATURE_NAMES:
            row[name] = 0.1 if label == "ATTACK" else -0.1
        row["unique_ip_count"] = 1
        row["unique_port_count"] = 1
        rows.append(row)
    table = pa.Table.from_pylist(rows, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_smoke_train_baselines_synthetic(tmp_path: Path) -> None:
    schema = tmp_path / "data" / "features" / "v1" / "feature_schema.json"
    write_feature_schema(schema)

    fit_id = "fit-aaaaaaaaaaaaaaaa"
    fit_ben_id = "fit-benign-dddddddddddddddd"
    val_atk = "val-attack-bbbbbbbbbbbbbbbb"
    val_ben = "val-benign-cccccccccccccccc"

    fit_dir = tmp_path / "data" / "modeling" / "v1" / "views" / "group_balanced" / "fit"
    _write_feature_parquet(fit_dir / f"{fit_id}.parquet", n=40, label="ATTACK", pcap_id=fit_id)
    _write_feature_parquet(
        fit_dir / f"{fit_ben_id}.parquet", n=40, label="BENIGN", pcap_id=fit_ben_id
    )

    val_atk_path = tmp_path / "data" / "features" / "v1" / "train" / f"{val_atk}.parquet"
    val_ben_path = tmp_path / "data" / "features" / "v1" / "train" / f"{val_ben}.parquet"
    _write_feature_parquet(val_atk_path, n=30, label="ATTACK", pcap_id=val_atk)
    _write_feature_parquet(val_ben_path, n=20, label="BENIGN", pcap_id=val_ben)

    fit_man = (
        tmp_path
        / "data"
        / "modeling"
        / "v1"
        / "views"
        / "group_balanced"
        / "fit_view_manifest.csv"
    )
    fit_man.parent.mkdir(parents=True, exist_ok=True)
    fit_man.write_text(
        "pcap_id,modeling_group_key,binary_label,attack_family,attack_type,"
        "benign_category,source_parquet_path,source_row_count,sampling_mode,"
        "group_budget,allocated_sample_rows,reservoir_seed,output_parquet_path,"
        "output_row_count,output_file_size,selection_sha256,status,resumed\n"
        f"{fit_id},DDoS|DDoS_ICMP,ATTACK,DDoS,DDoS_ICMP,,src,40,full,,40,1,"
        f"data/modeling/v1/views/group_balanced/fit/{fit_id}.parquet,40,1,abc,ok,false\n"
        f"{fit_ben_id},benign|singleton|Active,BENIGN,,,profiling_active,src,40,full,,40,1,"
        f"data/modeling/v1/views/group_balanced/fit/{fit_ben_id}.parquet,40,1,abc,ok,false\n",
        encoding="utf-8",
    )

    split = tmp_path / "data" / "modeling" / "v1" / "modeling_split_manifest.csv"
    split.parent.mkdir(parents=True, exist_ok=True)
    split.write_text(
        "pcap_path,pcap_id,modeling_group_key,modeling_split,binary_label,"
        "attack_family,attack_type,profiling_type,device,window_count,"
        "feature_parquet_path,selection_reason,benign_category,group_kind\n"
        f"raw/{fit_id}.pcap,{fit_id},DDoS|DDoS_ICMP,fit,ATTACK,DDoS,DDoS_ICMP,,,40,"
        f"data/features/v1/train/{fit_id}.parquet,fit,,attack_lineage\n"
        f"raw/{fit_ben_id}.pcap,{fit_ben_id},benign|singleton|Active,fit,BENIGN,,,,,40,"
        f"data/features/v1/train/{fit_ben_id}.parquet,fit,profiling_active,singleton\n"
        f"raw/{val_atk}.pcap,{val_atk},DDoS|DDoS_TCP,validation,ATTACK,DDoS,DDoS_TCP,,,30,"
        f"data/features/v1/train/{val_atk}.parquet,val,,attack_lineage\n"
        f"raw/{val_ben}.pcap,{val_ben},benign|singleton|Idle,validation,BENIGN,,,,,20,"
        f"data/features/v1/train/{val_ben}.parquet,val,profiling_idle,singleton\n",
        encoding="utf-8",
    )

    train_contract = tmp_path / "data" / "modeling" / "v1" / "training_view_contract.json"
    train_contract.write_text("{}\n", encoding="utf-8")

    complete = (
        tmp_path
        / "data"
        / "modeling"
        / "v1"
        / "views"
        / "group_balanced"
        / "fit_view_complete.json"
    )
    complete.write_text(
        json.dumps(
            {
                "status": "passed",
                "sampling_plan_id": "group_balanced",
                "modeling_split_strategy_version": "phase2a_v1",
                "validation_sampling": "never",
                "totals": {
                    "total_rows": 80,
                    "attack_rows": 40,
                    "benign_rows": 40,
                    "fit_pcaps": 2,
                    "validation_pcaps_touched": 0,
                    "test_pcaps_touched": 0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "data" / "modeling" / "v1" / "baselines" / "phase2b2_v1_smoke"
    result = train_baselines(
        smoke=True,
        project_root=tmp_path,
        output_dir=out,
        fit_complete_path=complete,
        fit_manifest_path=fit_man,
        split_manifest_path=split,
    )
    assert result.passed
    assert result.smoke_only is True
    assert result.test_pcaps_read == 0
    complete_payload = json.loads(result.run_complete_path.read_text(encoding="utf-8"))
    assert complete_payload["smoke_only"] is True
    assert complete_payload["status"] == "passed"
    assert (out / "comparison.csv").is_file()
    assert (out / "logistic_regression" / "metrics.json").is_file()
    assert (out / "hist_gradient_boosting" / "metrics.json").is_file()
    assert (out / "logistic_regression" / "models" / "logistic_regression.joblib").is_file()


def test_validation_never_uses_fit_view_path(tmp_path: Path) -> None:
    split = tmp_path / "split.csv"
    split.write_text(
        "pcap_path,pcap_id,modeling_group_key,modeling_split,binary_label,"
        "attack_family,attack_type,profiling_type,device,window_count,"
        "feature_parquet_path,selection_reason,benign_category,group_kind\n"
        "raw/a.pcap,a,DDoS|DDoS_TCP,validation,ATTACK,DDoS,DDoS_TCP,,,10,"
        "data/modeling/v1/views/group_balanced/fit/a.parquet,val,,attack_lineage\n",
        encoding="utf-8",
    )
    with pytest.raises(FeatureExtractionError, match="unsampled TRAIN"):
        load_validation_specs(split, project_root=tmp_path)
