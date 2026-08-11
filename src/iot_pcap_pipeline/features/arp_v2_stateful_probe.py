"""V2A A6: whole-PCAP ARP stateful feasibility probe (FIT only, no production extractor).

Streams ARP identity observations across each PCAP in capture order and asks
whether IP↔MAC conflicts become obvious once the artificial 25-packet window
boundary is removed. Identities stay internal; outputs are counts/ratios only.
No TEST access. No model training.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from iot_pcap_pipeline.features.arp_v2 import _valid_identity
from iot_pcap_pipeline.features.arp_v2_probe import (
    DEFAULT_ARP_PROBE_DIR,
    PROBE_GROUP_PROFILING_BENIGN,
    PROBE_GROUP_PUBLISHER_BENIGN,
    PROBE_GROUP_SPOOFING,
    load_arp_probe_targets,
)
from iot_pcap_pipeline.modeling.baselines.data import reject_test_path
from iot_pcap_pipeline.modeling.view import DEFAULT_SPLIT_MANIFEST_PATH
from iot_pcap_pipeline.paths import PROJECT_ROOT, to_repo_relative
from iot_pcap_pipeline.pcap.reader import iter_packets

ARP_STATEFUL_FEASIBILITY_VERSION = "v2a1_arp_stateful_feasibility"


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * (p / 100.0)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return xs[lo]
    weight = rank - lo
    return xs[lo] * (1.0 - weight) + xs[hi] * weight


def _dist_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "min": min(values),
        "max": max(values),
    }


@dataclass
class _IpState:
    macs: set[str] = field(default_factory=set)
    first_packet_index: int | None = None
    first_timestamp: float | None = None
    last_sha: str | None = None
    observation_count: int = 0
    conflict_declared: bool = False


@dataclass(frozen=True)
class PcapStatefulArpStats:
    """Whole-PCAP ARP conflict feasibility stats (no raw identities)."""

    packet_count: int
    arp_packet_count: int
    valid_identity_obs: int
    unique_sender_ip_count: int
    unique_sender_mac_count: int
    conflict_ip_count: int
    conflict_obs_count: int
    conflict_obs_ratio: float
    novel_mac_claim_count: int
    mapping_transition_count: int
    first_conflict_event_count: int
    packet_distance_to_first_conflict: dict[str, float | int]
    time_distance_to_first_conflict_seconds: dict[str, float | int]
    max_macs_per_sender_ip: int
    packet_distance_samples: tuple[float, ...] = ()
    time_distance_samples: tuple[float, ...] = ()

    def to_row(self) -> dict[str, Any]:
        pd = self.packet_distance_to_first_conflict
        td = self.time_distance_to_first_conflict_seconds
        return {
            "packet_count": self.packet_count,
            "arp_packet_count": self.arp_packet_count,
            "valid_identity_obs": self.valid_identity_obs,
            "unique_sender_ip_count": self.unique_sender_ip_count,
            "unique_sender_mac_count": self.unique_sender_mac_count,
            "conflict_ip_count": self.conflict_ip_count,
            "conflict_obs_count": self.conflict_obs_count,
            "conflict_obs_ratio": self.conflict_obs_ratio,
            "novel_mac_claim_count": self.novel_mac_claim_count,
            "mapping_transition_count": self.mapping_transition_count,
            "first_conflict_event_count": self.first_conflict_event_count,
            "max_macs_per_sender_ip": self.max_macs_per_sender_ip,
            "conflict_packet_distance_count": pd["count"],
            "conflict_packet_distance_mean": pd["mean"],
            "conflict_packet_distance_p50": pd["p50"],
            "conflict_packet_distance_p95": pd["p95"],
            "conflict_packet_distance_min": pd["min"],
            "conflict_packet_distance_max": pd["max"],
            "conflict_time_distance_count": td["count"],
            "conflict_time_distance_mean": td["mean"],
            "conflict_time_distance_p50": td["p50"],
            "conflict_time_distance_p95": td["p95"],
            "conflict_time_distance_min": td["min"],
            "conflict_time_distance_max": td["max"],
        }


def analyze_pcap_stateful_arp(pcap_path: Path) -> PcapStatefulArpStats:
    """Stream one PCAP and measure whole-capture ARP IP↔MAC conflict evidence."""
    reject_test_path(pcap_path)

    by_ip: dict[str, _IpState] = {}
    packet_count = 0
    arp_packet_count = 0
    valid_identity_obs = 0
    novel_mac_claim_count = 0
    mapping_transition_count = 0
    first_conflict_packet_distances: list[float] = []
    first_conflict_time_distances: list[float] = []
    all_macs: set[str] = set()
    observations: list[str] = []

    for packet in iter_packets(pcap_path):
        packet_count += 1
        if packet.is_arp:
            arp_packet_count += 1

        ident = _valid_identity(packet)
        if ident is None:
            continue

        spa, sha = ident
        valid_identity_obs += 1
        observations.append(spa)
        all_macs.add(sha)

        state = by_ip.get(spa)
        if state is None:
            by_ip[spa] = _IpState(
                macs={sha},
                first_packet_index=packet.packet_index,
                first_timestamp=packet.timestamp,
                last_sha=sha,
                observation_count=1,
            )
            continue

        state.observation_count += 1

        if state.last_sha is not None and sha != state.last_sha:
            mapping_transition_count += 1
        state.last_sha = sha

        if sha not in state.macs:
            novel_mac_claim_count += 1
            state.macs.add(sha)
            if not state.conflict_declared and len(state.macs) > 1:
                state.conflict_declared = True
                assert state.first_packet_index is not None
                assert state.first_timestamp is not None
                first_conflict_packet_distances.append(
                    float(packet.packet_index - state.first_packet_index)
                )
                first_conflict_time_distances.append(
                    float(packet.timestamp - state.first_timestamp)
                )

    conflict_ips = {ip for ip, st in by_ip.items() if len(st.macs) > 1}
    conflict_obs_count = sum(1 for spa in observations if spa in conflict_ips)
    conflict_obs_ratio = (
        conflict_obs_count / valid_identity_obs if valid_identity_obs else 0.0
    )
    max_macs = max((len(st.macs) for st in by_ip.values()), default=0)

    return PcapStatefulArpStats(
        packet_count=packet_count,
        arp_packet_count=arp_packet_count,
        valid_identity_obs=valid_identity_obs,
        unique_sender_ip_count=len(by_ip),
        unique_sender_mac_count=len(all_macs),
        conflict_ip_count=len(conflict_ips),
        conflict_obs_count=conflict_obs_count,
        conflict_obs_ratio=conflict_obs_ratio,
        novel_mac_claim_count=novel_mac_claim_count,
        mapping_transition_count=mapping_transition_count,
        first_conflict_event_count=len(first_conflict_packet_distances),
        packet_distance_to_first_conflict=_dist_stats(first_conflict_packet_distances),
        time_distance_to_first_conflict_seconds=_dist_stats(
            first_conflict_time_distances
        ),
        max_macs_per_sender_ip=max_macs,
        packet_distance_samples=tuple(first_conflict_packet_distances),
        time_distance_samples=tuple(first_conflict_time_distances),
    )


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate_group_rows(
    by_pcap: list[dict[str, Any]],
    *,
    group_packet_distances: dict[str, list[float]],
    group_time_distances: dict[str, list[float]],
) -> list[dict[str, Any]]:
    """Sum/pool group-level feasibility metrics from per-PCAP rows."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in by_pcap:
        groups[row["group"]].append(row)

    out: list[dict[str, Any]] = []
    for group, rows in sorted(groups.items()):
        valid = sum(int(r["valid_identity_obs"]) for r in rows)
        conflict_obs = sum(int(r["conflict_obs_count"]) for r in rows)
        conflict_ips = sum(int(r["conflict_ip_count"]) for r in rows)
        novel = sum(int(r["novel_mac_claim_count"]) for r in rows)
        transitions = sum(int(r["mapping_transition_count"]) for r in rows)
        first_events = sum(int(r["first_conflict_event_count"]) for r in rows)
        ratio = conflict_obs / valid if valid else 0.0
        pkt = _dist_stats(group_packet_distances.get(group, []))
        tim = _dist_stats(group_time_distances.get(group, []))
        out.append(
            {
                "group": group,
                "pcap_count": len(rows),
                "packet_count": sum(int(r["packet_count"]) for r in rows),
                "arp_packet_count": sum(int(r["arp_packet_count"]) for r in rows),
                "valid_identity_obs": valid,
                "unique_sender_ip_count_sum": sum(
                    int(r["unique_sender_ip_count"]) for r in rows
                ),
                "conflict_ip_count": conflict_ips,
                "conflict_obs_count": conflict_obs,
                "conflict_obs_ratio": ratio,
                "novel_mac_claim_count": novel,
                "mapping_transition_count": transitions,
                "first_conflict_event_count": first_events,
                "max_macs_per_sender_ip_max": max(
                    (int(r["max_macs_per_sender_ip"]) for r in rows), default=0
                ),
                "conflict_packet_distance_count": pkt["count"],
                "conflict_packet_distance_mean": pkt["mean"],
                "conflict_packet_distance_p50": pkt["p50"],
                "conflict_packet_distance_p95": pkt["p95"],
                "conflict_packet_distance_min": pkt["min"],
                "conflict_packet_distance_max": pkt["max"],
                "conflict_time_distance_count": tim["count"],
                "conflict_time_distance_mean": tim["mean"],
                "conflict_time_distance_p50": tim["p50"],
                "conflict_time_distance_p95": tim["p95"],
                "conflict_time_distance_min": tim["min"],
                "conflict_time_distance_max": tim["max"],
            }
        )
    return out


def run_arp_stateful_feasibility_probe(
    *,
    split_manifest_path: Path | None = None,
    output_dir: Path | None = None,
    project_root: Path | None = None,
    progress_file: TextIO | None = None,
) -> dict[str, Any]:
    """FIT-only whole-PCAP ARP conflict feasibility probe."""
    root = project_root or PROJECT_ROOT
    out_dir = Path(output_dir or DEFAULT_ARP_PROBE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = load_arp_probe_targets(split_manifest_path, project_root=root)
    if not targets:
        raise ValueError("no FIT ARP spoofing / benign PCAPs selected")

    by_pcap_rows: list[dict[str, Any]] = []
    group_packet_distances: dict[str, list[float]] = defaultdict(list)
    group_time_distances: dict[str, list[float]] = defaultdict(list)

    for idx, row in enumerate(targets, start=1):
        group = row["probe_group"]
        rel = row["pcap_path"]
        reject_test_path(rel)
        abs_path = root / rel
        if progress_file is not None:
            print(
                f"[{idx}/{len(targets)}] {group}: {rel}",
                file=progress_file,
                flush=True,
            )
        stats = analyze_pcap_stateful_arp(abs_path)
        group_packet_distances[group].extend(stats.packet_distance_samples)
        group_time_distances[group].extend(stats.time_distance_samples)

        rec = {
            "group": group,
            "pcap_id": row["pcap_id"],
            "pcap_path": rel,
            **stats.to_row(),
        }
        by_pcap_rows.append(rec)
        if progress_file is not None:
            print(
                f"  valid_arp={stats.valid_identity_obs} "
                f"conflict_ips={stats.conflict_ip_count} "
                f"conflict_obs_ratio={stats.conflict_obs_ratio:.4f} "
                f"novel_claims={stats.novel_mac_claim_count} "
                f"transitions={stats.mapping_transition_count}",
                file=progress_file,
                flush=True,
            )

    group_rows = _aggregate_group_rows(
        by_pcap_rows,
        group_packet_distances=group_packet_distances,
        group_time_distances=group_time_distances,
    )

    by_pcap_path = out_dir / "arp_stateful_by_pcap.csv"
    by_group_path = out_dir / "arp_stateful_by_group.csv"
    complete_path = out_dir / "arp_stateful_feasibility_complete.json"

    pcap_fields = list(by_pcap_rows[0].keys()) if by_pcap_rows else ["group"]
    group_fields = list(group_rows[0].keys()) if group_rows else ["group"]
    _write_csv(by_pcap_path, pcap_fields, by_pcap_rows)
    _write_csv(by_group_path, group_fields, group_rows)

    # Verdict helpers from group rollup.
    by_group = {r["group"]: r for r in group_rows}
    spoof = by_group.get(PROBE_GROUP_SPOOFING)
    pub = by_group.get(PROBE_GROUP_PUBLISHER_BENIGN)
    prof = by_group.get(PROBE_GROUP_PROFILING_BENIGN)

    def _ratio(row: dict[str, Any] | None) -> float:
        return float(row["conflict_obs_ratio"]) if row else 0.0

    spoof_ratio = _ratio(spoof)
    pub_ratio = _ratio(pub)
    prof_ratio = _ratio(prof)
    benign_max = max(pub_ratio, prof_ratio)
    spoof_conflict_ips = int(spoof["conflict_ip_count"]) if spoof else 0

    if spoof_ratio >= 0.70 and benign_max < 0.05:
        verdict = (
            "stateful_arp_strongly_justified: spoofing conflict_obs_ratio is high "
            "while benign remains near zero"
        )
        proceed = "consider_stateful_window_features"
    elif benign_max > spoof_ratio and spoof_conflict_ips <= 5:
        verdict = (
            "stop_arp_identity_features: whole-PCAP conflicts are not spoof-specific "
            f"(spoofing conflict_obs_ratio={spoof_ratio:.4f}, conflict_ips={spoof_conflict_ips}; "
            f"benign max conflict_obs_ratio={benign_max:.4f}). "
            "Capture does not expose a clean IP↔MAC conflict signal for detection."
        )
        proceed = "document_spoofing_as_limitation"
    elif spoof_ratio >= 0.20 and spoof_ratio > benign_max * 5:
        verdict = (
            "stateful_arp_promising_but_partial: spoofing shows elevated whole-PCAP "
            "conflicts vs benign; quantify coverage before feature engineering"
        )
        proceed = "consider_stateful_window_features_with_caution"
    elif spoof is not None and spoof_conflict_ips <= 5 and spoof_ratio < 0.05:
        verdict = (
            "stop_arp_identity_features: whole-PCAP analysis still shows only a "
            "handful of conflicting mappings — capture may not expose enough "
            "IP↔MAC transition evidence"
        )
        proceed = "document_spoofing_as_limitation"
    else:
        verdict = (
            "inconclusive_or_weak: whole-PCAP conflicts exist but do not cleanly "
            "separate spoofing from benign at high coverage"
        )
        proceed = "review_numbers_before_engineering"

    complete = {
        "status": "complete",
        "strategy_version": ARP_STATEFUL_FEASIBILITY_VERSION,
        "phase": "v2a1_a6",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Feasibility only: does removing the 25-packet memory boundary make "
            "ARP IP↔MAC conflicts obvious on FIT spoofing vs benign?"
        ),
        "production_extractor": False,
        "model_training": False,
        "data_access": {
            "development_data": "FIT only",
            "v1_final_test_access": False,
            "pcap_count": len(targets),
            "group_counts": {
                g: sum(1 for t in targets if t["probe_group"] == g)
                for g in (
                    PROBE_GROUP_SPOOFING,
                    PROBE_GROUP_PUBLISHER_BENIGN,
                    PROBE_GROUP_PROFILING_BENIGN,
                )
            },
        },
        "definitions": {
            "valid_identity_obs": (
                "ARP with valid IPv4 SPA, SPA != 0.0.0.0, and valid 6-byte SHA"
            ),
            "conflict_ip_count": "sender IPs that ever claim >1 distinct SHA",
            "conflict_obs_count": (
                "valid identity observations whose SPA is in the final conflict-IP set "
                "(retrospective; answers 'eventually involve an IP seen with another MAC')"
            ),
            "novel_mac_claim_count": (
                "new SHA first observed for an already-seen SPA (not every transition)"
            ),
            "mapping_transition_count": (
                "observations where SHA differs from the immediately previous SHA "
                "for that SPA (actual flip-flops)"
            ),
            "packet_distance_to_first_conflict": (
                "packet_index of first novel conflicting SHA minus packet_index of "
                "first observation of that SPA"
            ),
            "time_distance_to_first_conflict": (
                "timestamp delta for the same first-conflict event"
            ),
        },
        "group_summary": by_group,
        "verdict": verdict,
        "recommended_next_step": proceed,
        "artifacts": {
            "arp_stateful_by_pcap": to_repo_relative(by_pcap_path, project_root=root),
            "arp_stateful_by_group": to_repo_relative(by_group_path, project_root=root),
            "arp_stateful_feasibility_complete": to_repo_relative(
                complete_path, project_root=root
            ),
        },
    }
    complete_path.write_text(json.dumps(complete, indent=2) + "\n", encoding="utf-8")
    return complete


__all__ = [
    "ARP_STATEFUL_FEASIBILITY_VERSION",
    "PcapStatefulArpStats",
    "analyze_pcap_stateful_arp",
    "run_arp_stateful_feasibility_probe",
]
