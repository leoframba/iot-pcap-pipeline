"""Stable SHA-256 seed derivation for Phase 2A (never use built-in hash())."""

from __future__ import annotations

import hashlib

from iot_pcap_pipeline.paths import MODELING_SPLIT_STRATEGY_VERSION


def stable_seed_u64(
    salt: str,
    *,
    base_seed: int = 42,
    strategy_version: str = MODELING_SPLIT_STRATEGY_VERSION,
) -> int:
    """Return a process-stable 64-bit seed from version, base seed, and salt."""
    material = f"{strategy_version}|{base_seed}|{salt}".encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    return int(digest[:16], 16)


def reservoir_seed_for_pcap(
    pcap_id: str,
    *,
    base_seed: int = 42,
    strategy_version: str = MODELING_SPLIT_STRATEGY_VERSION,
) -> int:
    """Per-PCAP seed for the Phase 2B reservoir sampler contract."""
    return stable_seed_u64(
        pcap_id,
        base_seed=base_seed,
        strategy_version=strategy_version,
    )
