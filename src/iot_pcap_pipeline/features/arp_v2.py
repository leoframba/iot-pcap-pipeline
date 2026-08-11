"""V2A stateless ARP semantic features (experiment; not part of frozen V1).

Derives counts/ratios from ARP identity relationships inside one existing
25-packet window. Never writes raw MAC/IP strings as model features.
Does not mutate ``FeatureVector`` / the V1 27-feature extractor.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from iot_pcap_pipeline.pcap.packet import PacketRecord
from iot_pcap_pipeline.windowing.policy import WINDOW_SIZE
from iot_pcap_pipeline.windowing.window import PacketWindow

# Must match data/experiments/v2_arp/phase_v2a1/arp_feature_contract.json
ARP_V2_STRATEGY_VERSION = "v2a1_arp_stateless"

ARP_OP_REQUEST = 1
ARP_OP_REPLY = 2
_ZERO_IPV4 = "0.0.0.0"

ARP_V2_FEATURE_NAMES: tuple[str, ...] = (
    "arp_request_ratio",
    "arp_reply_ratio",
    "arp_probe_ratio",
    "arp_gratuitous_ratio",
    "arp_sender_ip_conflict_count",
    "arp_sender_ip_conflict_ratio",
    "arp_max_macs_per_sender_ip",
    "arp_mapping_change_count",
    "arp_eth_src_sha_mismatch_ratio",
    "arp_unique_sender_ip_count",
    "arp_unique_sender_mac_count",
)

assert len(ARP_V2_FEATURE_NAMES) == 11


@dataclass(frozen=True)
class ArpSemanticFeatures:
    """Candidate ARP semantic features for one packet window (V2A)."""

    arp_request_ratio: float
    arp_reply_ratio: float
    arp_probe_ratio: float
    arp_gratuitous_ratio: float
    arp_sender_ip_conflict_count: int
    arp_sender_ip_conflict_ratio: float
    arp_max_macs_per_sender_ip: int
    # Count of novel additional MAC claims for previously observed sender IPs
    # within the window (not every SPA→SHA transition / flip-flop).
    arp_mapping_change_count: int
    # mismatches / identity observations that also have a valid Ethernet src MAC
    arp_eth_src_sha_mismatch_ratio: float
    arp_unique_sender_ip_count: int
    arp_unique_sender_mac_count: int

    def to_ordered_values(self) -> tuple[float, ...]:
        data = asdict(self)
        return tuple(float(data[name]) for name in ARP_V2_FEATURE_NAMES)

    def to_feature_dict(self) -> dict[str, float | int]:
        return {name: getattr(self, name) for name in ARP_V2_FEATURE_NAMES}


def _arp_op(packet: PacketRecord) -> int | None:
    raw = packet.extra.get("arp_op")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _is_arp_probe(packet: PacketRecord) -> bool:
    """RFC 5227 ARP Probe: REQUEST with SPA == 0.0.0.0."""
    return _arp_op(packet) == ARP_OP_REQUEST and packet.src_ip == _ZERO_IPV4


def _is_arp_gratuitous(packet: PacketRecord) -> bool:
    """SPA == TPA announcement; excludes zero-SPA probes."""
    spa = packet.src_ip
    tpa = packet.dst_ip
    if spa is None or tpa is None:
        return False
    if spa == _ZERO_IPV4:
        return False
    return spa == tpa


def _valid_identity(packet: PacketRecord) -> tuple[str, str] | None:
    """Return (SPA, SHA) when usable for IP↔MAC consistency features.

    Excludes non-ARP, invalid/missing IPv4 SPA, SPA == 0.0.0.0 (probes), and
    invalid SHA (missing or non-6-byte → decoder stores None).
    """
    if not packet.is_arp:
        return None
    spa = packet.src_ip
    sha = packet.extra.get("arp_sha")
    if spa is None or spa == _ZERO_IPV4:
        return None
    if not isinstance(sha, str) or not sha:
        return None
    return spa, sha


def _empty_features() -> ArpSemanticFeatures:
    return ArpSemanticFeatures(
        arp_request_ratio=0.0,
        arp_reply_ratio=0.0,
        arp_probe_ratio=0.0,
        arp_gratuitous_ratio=0.0,
        arp_sender_ip_conflict_count=0,
        arp_sender_ip_conflict_ratio=0.0,
        arp_max_macs_per_sender_ip=0,
        arp_mapping_change_count=0,
        arp_eth_src_sha_mismatch_ratio=0.0,
        arp_unique_sender_ip_count=0,
        arp_unique_sender_mac_count=0,
    )


def extract_arp_semantic_features(window: PacketWindow) -> ArpSemanticFeatures:
    """Extract V2A ARP semantic features from one full 25-packet window (stateless).

    Requires exactly ``WINDOW_SIZE`` packets (frozen V1 policy). Ratios use
    ``n_arp`` (``is_arp`` packet count) as denominator and return 0.0 when the
    window has no ARP packets. Identity features ignore probes (SPA 0.0.0.0)
    and invalid SHA/SPA. ``arp_mapping_change_count`` counts novel additional
    MAC claims for previously observed sender IPs (not every transition).
    ``arp_eth_src_sha_mismatch_ratio`` denominates only identity observations
    that also have a valid Ethernet source MAC. No state across windows.
    """
    packets = window.packets
    n = len(packets)
    if n != WINDOW_SIZE:
        raise ValueError(
            f"V2A ARP windows must contain exactly {WINDOW_SIZE} packets, got {n}"
        )

    arp_packets = [p for p in packets if p.is_arp]
    n_arp = len(arp_packets)
    if n_arp == 0:
        return _empty_features()

    n_request = 0
    n_reply = 0
    n_probe = 0
    n_gratuitous = 0
    for packet in arp_packets:
        op = _arp_op(packet)
        if op == ARP_OP_REQUEST:
            n_request += 1
        elif op == ARP_OP_REPLY:
            n_reply += 1
        if _is_arp_probe(packet):
            n_probe += 1
        if _is_arp_gratuitous(packet):
            n_gratuitous += 1

    # Capture-order identity observations (valid SPA + SHA only).
    identities: list[tuple[str, str, PacketRecord]] = []
    for packet in packets:
        ident = _valid_identity(packet)
        if ident is None:
            continue
        identities.append((ident[0], ident[1], packet))

    n_ident = len(identities)
    macs_by_ip: dict[str, set[str]] = defaultdict(set)
    seen: dict[str, set[str]] = {}
    mapping_change_count = 0
    mismatch_eligible = 0
    mismatch_count = 0

    for spa, sha, packet in identities:
        macs_by_ip[spa].add(sha)
        # Novel additional MAC for an already-seen SPA (not flip-flop frequency).
        if spa not in seen:
            seen[spa] = {sha}
        elif sha not in seen[spa]:
            mapping_change_count += 1
            seen[spa].add(sha)

        src_mac = packet.extra.get("src_mac")
        if isinstance(src_mac, str) and src_mac:
            mismatch_eligible += 1
            if src_mac != sha:
                mismatch_count += 1

    conflict_ips = {ip for ip, macs in macs_by_ip.items() if len(macs) > 1}
    conflict_obs = sum(1 for spa, _sha, _p in identities if spa in conflict_ips)

    unique_ips = set(macs_by_ip)
    unique_macs: set[str] = set()
    for macs in macs_by_ip.values():
        unique_macs.update(macs)

    max_macs = max((len(macs) for macs in macs_by_ip.values()), default=0)

    return ArpSemanticFeatures(
        arp_request_ratio=n_request / n_arp,
        arp_reply_ratio=n_reply / n_arp,
        arp_probe_ratio=n_probe / n_arp,
        arp_gratuitous_ratio=n_gratuitous / n_arp,
        arp_sender_ip_conflict_count=len(conflict_ips),
        arp_sender_ip_conflict_ratio=(conflict_obs / n_ident) if n_ident else 0.0,
        arp_max_macs_per_sender_ip=max_macs,
        arp_mapping_change_count=mapping_change_count,
        arp_eth_src_sha_mismatch_ratio=(
            (mismatch_count / mismatch_eligible) if mismatch_eligible else 0.0
        ),
        arp_unique_sender_ip_count=len(unique_ips),
        arp_unique_sender_mac_count=len(unique_macs),
    )


def arp_v2_feature_contract_fragment() -> dict[str, Any]:
    """Serializable candidate-feature pin for the V2A experiment contract."""
    return {
        "strategy_version": ARP_V2_STRATEGY_VERSION,
        "feature_count": len(ARP_V2_FEATURE_NAMES),
        "feature_names": list(ARP_V2_FEATURE_NAMES),
        "denominator_basic_ratios": "n_arp_packets_in_window",
        "empty_arp_policy": "all_ratios_0_counts_0",
        "arp_probe_definition": {
            "op": "REQUEST",
            "spa": "0.0.0.0",
            "rfc": "5227",
        },
        "arp_gratuitous_definition": {
            "spa_equals_tpa": True,
            "exclude_spa": "0.0.0.0",
        },
        "identity_observation_excludes": [
            "spa_0.0.0.0",
            "invalid_or_missing_ipv4_spa",
            "invalid_or_missing_sha",
        ],
        "arp_mapping_change_count": (
            "count of novel additional MAC claims for previously observed "
            "sender IPs within the window"
        ),
        "arp_eth_src_sha_mismatch_denominator": (
            "valid ARP identity observations with a valid Ethernet source MAC"
        ),
        "required_window_size": WINDOW_SIZE,
        "raw_mac_model_features": False,
        "state_across_windows": False,
    }


__all__ = [
    "ARP_OP_REPLY",
    "ARP_OP_REQUEST",
    "ARP_V2_FEATURE_NAMES",
    "ARP_V2_STRATEGY_VERSION",
    "ArpSemanticFeatures",
    "arp_v2_feature_contract_fragment",
    "extract_arp_semantic_features",
]
