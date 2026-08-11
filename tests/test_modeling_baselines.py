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
    val_meta = complete_payload["validation"]
    assert val_meta["full_run_sampling"] == "never"
    assert val_meta["smoke_selection"] == "stratified_fixed_slice"
    assert val_meta["smoke_rows_per_group"] == 750
    assert val_meta["owltron_power"] == "all_available"
    assert "sampling" not in val_meta or val_meta.get("sampling") != "never"
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


def test_smoke_validation_budgets_cover_all_groups() -> None:
    from iot_pcap_pipeline.modeling.baselines.data import (
        ValidationPcapSpec,
        build_smoke_validation_budgets,
    )

    specs = [
        ValidationPcapSpec(
            pcap_id="ddos1",
            pcap_path="a",
            feature_parquet_path="data/features/v1/train/a.parquet",
            modeling_group_key="DDoS|DDoS_TCP",
            binary_label="ATTACK",
            attack_family="DDoS",
            attack_type="DDoS_TCP",
            benign_category="",
            window_count=10000,
        ),
        ValidationPcapSpec(
            pcap_id="dos1",
            pcap_path="b",
            feature_parquet_path="data/features/v1/train/b.parquet",
            modeling_group_key="DoS|DoS_TCP",
            binary_label="ATTACK",
            attack_family="DoS",
            attack_type="DoS_TCP",
            benign_category="",
            window_count=10000,
        ),
        ValidationPcapSpec(
            pcap_id="mqtt1",
            pcap_path="c",
            feature_parquet_path="data/features/v1/train/c.parquet",
            modeling_group_key="MQTT|MQTT_DoS_Publish_Flood",
            binary_label="ATTACK",
            attack_family="MQTT",
            attack_type="MQTT_DoS_Publish_Flood",
            benign_category="",
            window_count=10000,
        ),
        ValidationPcapSpec(
            pcap_id="recon1",
            pcap_path="d",
            feature_parquet_path="data/features/v1/train/d.parquet",
            modeling_group_key="Recon|OS_Scan",
            binary_label="ATTACK",
            attack_family="Recon",
            attack_type="OS_Scan",
            benign_category="",
            window_count=10000,
        ),
        ValidationPcapSpec(
            pcap_id="idle1",
            pcap_path="e",
            feature_parquet_path="data/features/v1/train/e.parquet",
            modeling_group_key="benign|singleton|Idle",
            binary_label="BENIGN",
            attack_family="",
            attack_type="",
            benign_category="profiling_idle",
            window_count=13149,
        ),
        ValidationPcapSpec(
            pcap_id="owl-int",
            pcap_path="f",
            feature_parquet_path="data/features/v1/train/f.parquet",
            modeling_group_key="benign|device|Owltron_Camera",
            binary_label="BENIGN",
            attack_family="",
            attack_type="",
            benign_category="profiling_interaction",
            window_count=9315,
        ),
        ValidationPcapSpec(
            pcap_id="owl-pwr",
            pcap_path="g",
            feature_parquet_path="data/features/v1/train/g.parquet",
            modeling_group_key="benign|device|Owltron_Camera",
            binary_label="BENIGN",
            attack_family="",
            attack_type="",
            benign_category="profiling_power",
            window_count=40,
        ),
    ]
    budgets = build_smoke_validation_budgets(specs, rows_per_group=750)
    assert budgets["ddos1"] == 750
    assert budgets["dos1"] == 750
    assert budgets["mqtt1"] == 750
    assert budgets["recon1"] == 750
    assert budgets["idle1"] == 750
    assert budgets["owl-int"] == 750
    assert budgets["owl-pwr"] == 40  # keep all
    assert sum(budgets.values()) == 750 * 6 + 40


def test_prepare_and_full_train_requires_frozen_contract(tmp_path: Path) -> None:
    from iot_pcap_pipeline.modeling.baselines.contract import (
        load_frozen_baseline_contract,
        prepare_baseline_run,
    )
    from iot_pcap_pipeline.modeling.view import file_sha256
    from iot_pcap_pipeline.features.parquet import feature_schema_sha256

    # Minimal tree under tmp
    schema = tmp_path / "data" / "features" / "v1" / "feature_schema.json"
    write_feature_schema(schema)
    split = tmp_path / "data" / "modeling" / "v1" / "modeling_split_manifest.csv"
    split.parent.mkdir(parents=True, exist_ok=True)
    split.write_text("pcap_path\n", encoding="utf-8")
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
    fit_man.write_text("pcap_id\n", encoding="utf-8")
    train_c = tmp_path / "data" / "modeling" / "v1" / "training_view_contract.json"
    train_c.write_text("{}\n", encoding="utf-8")
    complete = fit_man.parent / "fit_view_complete.json"
    # prepare requires full totals — use smoke_only path via direct write instead
    with pytest.raises(FeatureExtractionError):
        load_frozen_baseline_contract(
            tmp_path / "missing.json", project_root=tmp_path
        )

    # Write a frozen contract manually and verify load + hash refuse on drift
    from iot_pcap_pipeline.modeling.baselines.contract import write_baseline_contract

    out = tmp_path / "data" / "modeling" / "v1" / "baselines" / "phase2b2_v1" / "baseline_contract.json"
    write_baseline_contract(
        out,
        project_root=tmp_path,
        smoke_only=False,
        status="frozen",
        fit_view_manifest_path=fit_man,
        split_manifest_path=split,
        training_view_contract_path=train_c,
        feature_schema_path=schema,
    )
    loaded = load_frozen_baseline_contract(out, project_root=tmp_path)
    assert loaded["status"] == "frozen"
    assert loaded["smoke_only"] is False
    split.write_text("pcap_path\nx\n", encoding="utf-8")
    with pytest.raises(FeatureExtractionError, match="mismatch"):
        load_frozen_baseline_contract(out, project_root=tmp_path)


def test_threshold_sweep_math_synthetic() -> None:
    from iot_pcap_pipeline.modeling.baselines.threshold_sweep import (
        ValidationScoreTape,
        metrics_at_threshold,
        sweep_model,
        threshold_for_benign_fpr,
    )

    # 100 benign scores in [0,1), 100 attack scores mostly high with a few low.
    benign = np.linspace(0.0, 0.99, 100, dtype=np.float32)
    attack = np.array([0.05] * 10 + [0.99] * 90, dtype=np.float32)
    y = np.array([0] * 100 + [1] * 100, dtype=np.uint8)
    scores = np.concatenate([benign, attack])
    # codes: idle=10 for benign, recon=4 for attack
    g = np.array([10] * 100 + [4] * 100, dtype=np.uint8)
    tape = ValidationScoreTape(y_true=y, scores=scores, group_code=g)

    row = metrics_at_threshold(
        tape, threshold=0.5, model_id="toy", point_type="fixed_threshold"
    )
    assert row["benign_fp"] == int(np.sum(benign >= 0.5))
    assert row["recon_os_scan_recall"] == pytest.approx(0.9)

    thr = threshold_for_benign_fpr(tape, 0.10)
    fpr_row = metrics_at_threshold(
        tape,
        threshold=thr,
        model_id="toy",
        point_type="fpr_target",
        fpr_target=0.10,
    )
    assert fpr_row["benign_fpr"] <= 0.10 + 1e-9

    fixed, fpr_rows = sweep_model(tape, model_id="toy")
    assert len(fixed) >= 11
    assert len(fpr_rows) == 6
