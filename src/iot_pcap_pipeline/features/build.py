"""Gate-B TRAIN smoke extraction orchestration."""

from __future__ import annotations

import csv
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from iot_pcap_pipeline.features.characterize import write_characterization_csv
from iot_pcap_pipeline.features.extractor import extract_features
from iot_pcap_pipeline.features.schema import (
    METADATA_COLUMN_NAMES,
    V1_FEATURE_NAMES,
    write_feature_schema,
)
from iot_pcap_pipeline.features.validate import (
    FeatureInvariantError,
    validate_window_and_features,
)
from iot_pcap_pipeline.paths import (
    DEFAULT_FEATURES_DIR,
    DEFAULT_MANIFEST_DIR,
    FEATURE_STRATEGY_VERSION,
    PROJECT_ROOT,
    to_repo_relative,
)
from iot_pcap_pipeline.pcap.reader import iter_packets
from iot_pcap_pipeline.windowing.policy import frozen_window_policy
from iot_pcap_pipeline.windowing.stream import (
    FeatureExtractionError,
    WindowStreamStats,
    iter_windows,
)

DEFAULT_SMOKE_DIR = DEFAULT_FEATURES_DIR / "v1" / "smoke"
DEFAULT_SMOKE_FEATURES_CSV = DEFAULT_SMOKE_DIR / "train_features_smoke.csv"
DEFAULT_SMOKE_CHARACTERIZATION_CSV = (
    DEFAULT_SMOKE_DIR / "training_feature_characterization.csv"
)
DEFAULT_MAX_WINDOWS_PER_PCAP = 10_000

# Representative TRAIN smoke set (repo-relative paths).
# Must be inventory split=train only (SenseU device group is TEST).
DEFAULT_SMOKE_PCAP_PATHS: tuple[str, ...] = (
    "data/raw/WiFI_and_MQTT/attacks/pcap/train/Benign_train.pcap",
    "data/raw/WiFI_and_MQTT/profiling/PCAP/Idle/Idle.pcap",
    "data/raw/WiFI_and_MQTT/profiling/PCAP/Active/Active.pcap",
    "data/raw/WiFI_and_MQTT/profiling/PCAP/Broker/ActiveBroker.pcap",
    "data/raw/WiFI_and_MQTT/profiling/PCAP/Power/Blink_Mini_Camera_Power.pcap",
    "data/raw/WiFI_and_MQTT/profiling/PCAP/Interactions/Blink_Camera/Blink_Camera_LAN_WATCH.pcap",
    "data/raw/WiFI_and_MQTT/profiling/PCAP/Interactions/Singcall/Singcall_LAN_PHYSICAL.pcap",
    "data/raw/WiFI_and_MQTT/profiling/PCAP/Interactions/M1T_Camera/M1T_Camera_WAN_RECORDING.pcap",
    "data/raw/WiFI_and_MQTT/attacks/pcap/train/TCP_IP-DDoS-SYN2_train.pcap",
    "data/raw/WiFI_and_MQTT/attacks/pcap/train/TCP_IP-DoS-SYN1_train.pcap",
    "data/raw/WiFI_and_MQTT/attacks/pcap/train/MQTT-DDoS-Connect_Flood_train.pcap",
    "data/raw/WiFI_and_MQTT/attacks/pcap/train/Recon-Port_Scan_train.pcap",
    "data/raw/WiFI_and_MQTT/attacks/pcap/train/ARP_Spoofing_train.pcap",
)


@dataclass
class ExtractionAccount:
    pcaps_processed: int = 0
    pcaps_failed: int = 0
    packets_processed: int = 0
    segments: int = 0
    full_windows: int = 0
    dropped_partial_packets: int = 0
    zero_span_windows: int = 0
    invariant_failures: int = 0
    by_parse_status: Counter[str] = field(default_factory=Counter)
    windows_by_category: Counter[str] = field(default_factory=Counter)
    elapsed_seconds: float = 0.0


def load_inventory_index(inventory_path: Path) -> dict[str, dict[str, str]]:
    if not inventory_path.is_file():
        return {}
    with inventory_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {r["pcap_path"]: r for r in rows}


def _category_from_meta(meta: dict[str, Any]) -> str:
    profiling = (meta.get("profiling_type") or "").strip()
    family = (meta.get("attack_family") or "").strip()
    label = (meta.get("binary_label") or "").strip()
    if profiling:
        return f"profiling_{profiling}"
    if family:
        return family
    if label == "BENIGN":
        return "publisher_benign"
    if label:
        return label
    return "unknown"


def extract_pcap_feature_rows(
    pcap_path: Path,
    *,
    meta: dict[str, Any] | None = None,
    max_windows: int | None = None,
    project_root: Path | None = None,
    validate: bool = True,
) -> tuple[list[dict[str, Any]], WindowStreamStats]:
    """Extract feature rows from one PCAP using the shared production path."""
    root = project_root or PROJECT_ROOT
    policy = frozen_window_policy()
    stats = WindowStreamStats()
    rel = to_repo_relative(pcap_path, project_root=root)
    base_meta = {
        "pcap_path": rel,
        "split": "",
        "binary_label": "",
        "attack_family": "",
        "attack_type": "",
        "profiling_type": "",
        "profiling_variant": "",
        "device": "",
        "capture_session": "",
        "feature_strategy_version": FEATURE_STRATEGY_VERSION,
    }
    if meta:
        for key in base_meta:
            if key in meta and meta[key] not in (None, ""):
                base_meta[key] = meta[key]

    rows: list[dict[str, Any]] = []
    for window in iter_windows(
        iter_packets(pcap_path),
        policy,
        stats=stats,
        max_windows=max_windows,
    ):
        features = extract_features(window)
        if validate:
            validate_window_and_features(window, features)
        row = {
            **base_meta,
            "segment_index": window.segment_index,
            "window_index": window.window_index,
            "packet_index_start": window.packet_index_start,
            "packet_index_end": window.packet_index_end,
            **features.to_feature_dict(),
        }
        rows.append(row)
    return rows, stats


def run_smoke_extraction(
    *,
    pcap_paths: list[Path] | None = None,
    inventory_path: Path | str | None = None,
    output_path: Path | str | None = None,
    characterization_path: Path | str | None = None,
    schema_path: Path | str | None = None,
    max_windows_per_pcap: int | None = DEFAULT_MAX_WINDOWS_PER_PCAP,
    project_root: Path | None = None,
    progress_file: TextIO | None = None,
) -> dict[str, Any]:
    """TRAIN smoke build: schema + features CSV + characterization CSV."""
    root = (project_root or PROJECT_ROOT).resolve()
    inv = Path(inventory_path or (DEFAULT_MANIFEST_DIR / "pcap_inventory.csv"))
    if not inv.is_absolute():
        inv = root / inv
    index = load_inventory_index(inv)

    if pcap_paths is None:
        paths = []
        for rel in DEFAULT_SMOKE_PCAP_PATHS:
            meta = index.get(rel, {})
            split = meta.get("split")
            if split and split != "train":
                raise FeatureExtractionError(
                    f"DEFAULT_SMOKE_PCAP_PATHS contains non-TRAIN PCAP: {rel} "
                    f"(split={split!r}). Update the smoke list."
                )
            if index and not meta:
                raise FeatureExtractionError(
                    f"DEFAULT_SMOKE_PCAP_PATHS path missing from inventory: {rel}"
                )
            paths.append(root / rel)
    else:
        paths = [
            p if p.is_absolute() else (root / p) for p in pcap_paths
        ]

    features_out = Path(output_path or DEFAULT_SMOKE_FEATURES_CSV)
    if not features_out.is_absolute():
        features_out = root / features_out
    char_out = Path(characterization_path or DEFAULT_SMOKE_CHARACTERIZATION_CSV)
    if not char_out.is_absolute():
        char_out = root / char_out
    schema_out = write_feature_schema(schema_path)

    account = ExtractionAccount()
    all_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for path in paths:
        rel = to_repo_relative(path, project_root=root)
        if progress_file is not None:
            print(f"Extracting {rel} ...", file=progress_file, flush=True)
        meta = dict(index.get(rel, {}))
        # Ensure TRAIN-only smoke does not silently include TEST inventory rows.
        if meta.get("split") and meta.get("split") != "train":
            raise FeatureExtractionError(
                f"smoke extraction refused non-TRAIN inventory row: {rel} "
                f"(split={meta.get('split')!r})"
            )
        try:
            rows, stats = extract_pcap_feature_rows(
                path,
                meta=meta,
                max_windows=max_windows_per_pcap,
                project_root=root,
                validate=True,
            )
        except (FeatureExtractionError, FeatureInvariantError, FileNotFoundError) as exc:
            account.pcaps_failed += 1
            if progress_file is not None:
                print(f"FAILED {rel}: {exc}", file=progress_file, flush=True)
            raise

        account.pcaps_processed += 1
        account.packets_processed += stats.packets_seen
        account.segments += stats.segment_count
        account.full_windows += stats.full_window_count
        account.dropped_partial_packets += stats.dropped_partial_packet_count
        account.by_parse_status.update(stats.by_parse_status)
        cat = _category_from_meta(meta if meta else {"binary_label": ""})
        account.windows_by_category[cat] += len(rows)
        for row in rows:
            if float(row["window_span_seconds"]) == 0.0:
                account.zero_span_windows += 1
        all_rows.extend(rows)

    account.elapsed_seconds = time.perf_counter() - t0

    columns = list(METADATA_COLUMN_NAMES) + list(V1_FEATURE_NAMES)
    features_out.parent.mkdir(parents=True, exist_ok=True)
    with features_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow({c: row.get(c, "") for c in columns})

    write_characterization_csv(all_rows, char_out)

    return {
        "rows": all_rows,
        "account": account,
        "features_path": features_out,
        "characterization_path": char_out,
        "schema_path": schema_out,
        "max_windows_per_pcap": max_windows_per_pcap,
    }


def format_smoke_summary(result: dict[str, Any]) -> str:
    account: ExtractionAccount = result["account"]
    capped = result.get("max_windows_per_pcap")
    lines = [
        "Phase 1C.2 Gate B — TRAIN smoke feature extraction",
        "=" * 60,
        f"feature_strategy_version: {FEATURE_STRATEGY_VERSION}",
        f"window_policy: {frozen_window_policy().config_id}",
        f"PCAPs processed: {account.pcaps_processed}",
        f"PCAPs failed: {account.pcaps_failed}",
        f"packets processed: {account.packets_processed:,}",
        f"segments: {account.segments:,}",
        f"full windows emitted: {account.full_windows:,}",
        f"partial packets dropped: {account.dropped_partial_packets:,}",
        f"zero-span windows: {account.zero_span_windows:,}",
        f"invariant failures: {account.invariant_failures}",
        f"elapsed_s: {account.elapsed_seconds:.3f}",
    ]
    if account.elapsed_seconds > 0 and account.packets_processed > 0:
        lines.append(
            f"throughput: {account.packets_processed / account.elapsed_seconds:,.0f} packets/s  "
            f"{account.full_windows / account.elapsed_seconds:,.1f} windows/s"
        )
    if capped is not None:
        lines.append(f"max_windows_per_pcap (capped): {capped:,}")
    lines.append("windows by category:")
    for cat, count in sorted(account.windows_by_category.items()):
        lines.append(f"  {cat}: {count:,}")
    lines.append("parse_status counts:")
    for status in ("ok", "unsupported", "partial", "malformed", "error"):
        lines.append(f"  {status}: {account.by_parse_status.get(status, 0):,}")
    lines.extend(
        [
            "",
            f"Wrote {result['schema_path']}",
            f"Wrote {result['features_path']}",
            f"Wrote {result['characterization_path']}",
            "",
            "GATE B — PASSED",
            f"Frozen feature contract: {FEATURE_STRATEGY_VERSION} (27 features).",
            (
                "tcp_urg_ratio retained; constant-feature drop deferred to post-TRAIN "
                "contract review (TEST must not decide)."
            ),
            "Phase 1C.3: stream to Parquet (do not list-accumulate all windows).",
            "Do not process TEST until 1C.3 TEST extraction is explicitly started.",
        ]
    )
    return "\n".join(lines)
