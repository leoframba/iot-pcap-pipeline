"""D1 classify_pcap + serving runtime isolation tests."""

from __future__ import annotations

import sys
from pathlib import Path

import dpkt
from pcap_synth import eth_ip_tcp, write_pcap

from iot_pcap_pipeline.pcap.decode import DLT_EN10MB
from iot_pcap_pipeline.serving.classify import classify_pcap
from iot_pcap_pipeline.serving.errors import (
    STATUS_INSUFFICIENT_DATA,
    STATUS_INVALID_INPUT,
    STATUS_OK,
    STATUS_UNSUPPORTED_INPUT,
)
from iot_pcap_pipeline.serving.labels import ATTACK_CLASS, BENIGN_CLASS
from iot_pcap_pipeline.serving.model import V1InferenceEngine
from iot_pcap_pipeline.windowing.policy import WINDOW_SIZE


def _ethernet_pcap(path: Path, n_packets: int) -> Path:
    packets = [
        (float(i) * 0.001, eth_ip_tcp(sport=1000 + i, flags=dpkt.tcp.TH_SYN))
        for i in range(n_packets)
    ]
    return write_pcap(path, packets, linktype=DLT_EN10MB)


def test_serving_contract_import_avoids_research_stack() -> None:
    banned = (
        "pyarrow",
        "iot_pcap_pipeline.features.parquet",
        "iot_pcap_pipeline.modeling.baselines.model_input",
        "iot_pcap_pipeline.modeling.baselines.phase2c_freeze",
        "iot_pcap_pipeline.modeling.view",
        "iot_pcap_pipeline.cli",
    )
    src = Path("src/iot_pcap_pipeline/serving/contract.py").read_text(encoding="utf-8")
    for name in banned:
        assert name not in src

    # Clean interpreter: contract verify must not pull research/storage modules.
    import subprocess

    script = r"""
import sys
from iot_pcap_pipeline.serving.contract import verify_serving_contract
verify_serving_contract()
banned = [
    "pyarrow",
    "iot_pcap_pipeline.features.parquet",
    "iot_pcap_pipeline.modeling.baselines.model_input",
    "iot_pcap_pipeline.modeling.baselines.phase2c_freeze",
    "iot_pcap_pipeline.modeling.view",
    "iot_pcap_pipeline.cli",
]
leaked = [m for m in banned if m in sys.modules]
assert not leaked, leaked
print("ok")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout


def test_serving_labels_match_research_label_mapping() -> None:
    from iot_pcap_pipeline.modeling.baselines.constants import LABEL_MAPPING

    assert BENIGN_CLASS == LABEL_MAPPING["BENIGN"] == 0
    assert ATTACK_CLASS == LABEL_MAPPING["ATTACK"] == 1


def test_engine_loads_default() -> None:
    engine = V1InferenceEngine.load_default()
    assert engine.model_sha256.startswith("c07ef408")
    assert len(engine.feature_names) == 22
    assert int(engine.estimator.n_features_in_) == 22


def test_classify_insufficient_for_short_pcap(tmp_path: Path) -> None:
    engine = V1InferenceEngine.load_default()
    path = _ethernet_pcap(tmp_path / "short.pcap", n_packets=WINDOW_SIZE)  # 1 window
    result = classify_pcap(path, engine=engine, batch_size=8)
    assert result.status == STATUS_INSUFFICIENT_DATA
    assert result.prediction is None
    assert result.window_summary["total_windows"] == 1


def test_classify_ok_for_three_windows(tmp_path: Path) -> None:
    engine = V1InferenceEngine.load_default()
    path = _ethernet_pcap(tmp_path / "ok.pcap", n_packets=WINDOW_SIZE * 3)
    result = classify_pcap(path, engine=engine, batch_size=2)
    assert result.status == STATUS_OK
    assert result.prediction in {"ATTACK", "BENIGN"}
    assert result.window_summary["total_windows"] == 3
    assert result.pcap_attack_score is not None
    assert 0.0 <= result.pcap_attack_score <= 1.0
    assert "probability" not in result.to_dict()["model"]["score_semantics"]


def test_classify_rejects_non_ethernet_linktype(tmp_path: Path) -> None:
    engine = V1InferenceEngine.load_default()
    ip = dpkt.ip.IP(
        src=b"\n\x00\x00\x01",
        dst=b"\n\x00\x00\x02",
        p=dpkt.ip.IP_PROTO_TCP,
        data=dpkt.tcp.TCP(sport=1, dport=80, flags=dpkt.tcp.TH_SYN),
    )
    ip.len = len(ip)
    path = write_pcap(
        tmp_path / "raw.pcap",
        [(0.0, bytes(ip))] * (WINDOW_SIZE * 3),
        linktype=101,  # DLT_RAW / LINKTYPE_RAW
    )
    result = classify_pcap(path, engine=engine)
    assert result.status == STATUS_UNSUPPORTED_INPUT
    assert result.prediction is None


def test_classify_invalid_not_pcap(tmp_path: Path) -> None:
    engine = V1InferenceEngine.load_default()
    path = tmp_path / "not.pcap"
    path.write_bytes(b"this is not a pcap")
    result = classify_pcap(path, engine=engine)
    assert result.status == STATUS_INVALID_INPUT
    assert result.prediction is None


def test_classify_missing_file(tmp_path: Path) -> None:
    engine = V1InferenceEngine.load_default()
    result = classify_pcap(tmp_path / "missing.pcap", engine=engine)
    assert result.status == STATUS_INVALID_INPUT
