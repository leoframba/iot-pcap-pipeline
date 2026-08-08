"""Shared fixtures for Phase 1A tests."""

from __future__ import annotations

from pathlib import Path

import pytest


def _touch(path: Path, size: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


@pytest.fixture
def synthetic_raw(tmp_path: Path) -> Path:
    """Minimal CICIoMT-like tree covering attacks + profiling cases."""
    raw = tmp_path / "WiFI_and_MQTT"

    attacks_train = raw / "attacks" / "pcap" / "train"
    attacks_test = raw / "attacks" / "pcap" / "test"
    _touch(attacks_train / "TCP_IP-DDoS-UDP1_train.pcap", 100)
    _touch(attacks_train / "TCP_IP-DDoS-UDP2_train.pcap", 110)
    _touch(attacks_train / "Benign_train.pcap", 50)
    _touch(attacks_train / "Recon-Port_Scan_train.pcap", 80)
    _touch(attacks_train / "ARP_Spoofing_train.pcap", 70)
    _touch(attacks_test / "TCP_IP-DDoS-UDP1_test.pcap", 90)
    _touch(attacks_test / "Benign_test.pcap", 40)
    _touch(attacks_test / "MQTT-Malformed_Data_test.pcap", 60)

    # Contradictory split signal
    _touch(attacks_train / "Recon-OS_Scan_test.pcap", 30)

    profiling = raw / "profiling" / "PCAP"
    _touch(profiling / "Idle" / "Idle.pcap", 200)
    _touch(profiling / "Active" / "Active.pcap", 210)
    _touch(profiling / "Broker" / "ActiveBroker.pcap", 220)

    # Device A: 1 power + 2 interactions = 3
    _touch(profiling / "Power" / "Blink_Mini_Camera_Power.pcap", 15)
    blink = profiling / "Interactions" / "Blink_Camera"
    _touch(blink / "Blink_Camera_LAN_MIC.pcap", 16)
    _touch(blink / "Blink_Camera_WAN_MIC.pcap", 17)

    # Device B: 1 power + 2 interactions = 3
    _touch(profiling / "Power" / "SenseUBaby_Power.pcap", 18)
    sense = profiling / "Interactions" / "SenseU"
    _touch(sense / "SenseU_LAN_EMERGENCY.pcap", 19)
    _touch(sense / "SenseU_WAN_EMERGENCY.pcap", 20)

    # Device C: 1 power + 2 interactions = 3
    _touch(profiling / "Power" / "Singcall_Power.pcap", 21)
    sing = profiling / "Interactions" / "Singcall"
    _touch(sing / "Singcall_LAN_PHYSICAL.pcap", 22)
    _touch(sing / "Singcall_WAN_PHYSICAL.pcap", 23)

    # Unknown device spelling should be unresolved
    _touch(profiling / "Power" / "UnknownGadget_Power.pcap", 24)

    # Preserve typo token
    m1t = profiling / "Interactions" / "M1T_Camera"
    _touch(m1t / "M1T_Camera_LAN_PRECORDING.pcap", 25)
    _touch(profiling / "Power" / "M1T_Camera_Power.pcap", 26)

    return raw
