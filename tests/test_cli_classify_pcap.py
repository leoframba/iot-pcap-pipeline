"""CLI smoke for classify-pcap."""

from __future__ import annotations

from pathlib import Path

import dpkt
from pcap_synth import eth_ip_tcp, write_pcap

from iot_pcap_pipeline.cli import main
from iot_pcap_pipeline.windowing.policy import WINDOW_SIZE


def test_classify_pcap_cli_prints_json(tmp_path: Path, capsys) -> None:
    path = tmp_path / "smoke.pcap"
    packets = [
        (float(i) * 0.001, eth_ip_tcp(sport=3000 + i, flags=dpkt.tcp.TH_SYN))
        for i in range(WINDOW_SIZE * 3)
    ]
    write_pcap(path, packets, linktype=1)
    code = main(["classify-pcap", str(path), "--batch-size", "2"])
    assert code == 0
    out = capsys.readouterr().out
    assert '"status": "OK"' in out
    assert '"prediction"' in out
    assert '"model_version": "v1_hgb22_nontemporal"' in out
    assert "model_artifact_sha256" not in out


def test_classify_pcap_cli_invalid_exit_code(tmp_path: Path) -> None:
    path = tmp_path / "bad.pcap"
    path.write_bytes(b"not a pcap")
    assert main(["classify-pcap", str(path)]) == 1
