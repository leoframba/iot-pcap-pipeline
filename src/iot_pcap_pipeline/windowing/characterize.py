"""Phase 1C.1 TRAIN-only windowing-policy characterization.

Timestamp-only: reuses ``iter_timestamps`` and never decodes frames.
One PCAP is scanned once while all candidate policies are updated concurrently.
"""

from __future__ import annotations

import csv
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from iot_pcap_pipeline.paths import (
    DEFAULT_FEATURES_DIR,
    DEFAULT_MANIFEST_DIR,
    PROJECT_ROOT,
    to_repo_relative,
)
from iot_pcap_pipeline.pcap.timestamps import iter_timestamps
from iot_pcap_pipeline.windowing.policy import (
    DEFAULT_BACKWARD_RESET_SECONDS,
    WINDOWING_STRATEGY_VERSION,
    WindowPolicy,
    candidate_policies,
)

DEFAULT_CHARACTERIZATION_CSV = (
    DEFAULT_FEATURES_DIR / "windowing_characterization_train.csv"
)
DEFAULT_WORKERS = 4

CHARACTERIZATION_COLUMNS: list[str] = [
    "pcap_path",
    "source",
    "binary_label",
    "attack_family",
    "attack_type",
    "profiling_type",
    "profiling_variant",
    "device",
    "capture_session",
    "window_size",
    "inactivity_timeout_seconds",
    "backward_reset_seconds",
    "packet_count",
    "positive_gap_boundary_count",
    "backward_discontinuity_boundary_count",
    "segment_count",
    "full_window_count",
    "emitted_packet_count",
    "dropped_partial_window_count",
    "dropped_partial_packet_count",
    "packet_retention_ratio",
    "zero_span_window_count",
    "zero_span_window_ratio",
    "window_span_min",
    "window_span_mean",
    "window_span_p50",
    "window_span_p95",
    "window_span_p99",
    "window_span_max",
    "windowing_strategy_version",
]

# Aggregate summary buckets (analysis only; never feed segmentation).
SUMMARY_CATEGORY_ORDER: tuple[str, ...] = (
    "publisher_benign",
    "profiling_idle",
    "profiling_active",
    "profiling_power",
    "profiling_interaction",
    "DDoS",
    "DoS",
    "MQTT",
    "Recon",
    "Spoofing",
)


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
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


@dataclass
class PolicyAccumulator:
    """Streaming state for one windowing policy over one PCAP."""

    policy: WindowPolicy
    packet_count: int = 0
    positive_gap_boundary_count: int = 0
    backward_discontinuity_boundary_count: int = 0
    segment_count: int = 0
    full_window_count: int = 0
    emitted_packet_count: int = 0
    dropped_partial_window_count: int = 0
    dropped_partial_packet_count: int = 0
    zero_span_window_count: int = 0
    window_spans: list[float] = field(default_factory=list)
    _prev_ts: float | None = field(default=None, repr=False)
    _win_count: int = field(default=0, repr=False)
    _win_first_ts: float | None = field(default=None, repr=False)
    _win_last_ts: float | None = field(default=None, repr=False)

    def observe(self, timestamp: float) -> None:
        """Consume one packet timestamp in original capture order."""
        self.packet_count += 1
        if self._prev_ts is None:
            self._start_segment(timestamp)
            self._prev_ts = timestamp
            return

        delta = timestamp - self._prev_ts
        if delta > self.policy.inactivity_timeout_seconds:
            self._close_partial()
            self.positive_gap_boundary_count += 1
            self._start_segment(timestamp)
        elif delta < -self.policy.backward_reset_seconds:
            self._close_partial()
            self.backward_discontinuity_boundary_count += 1
            self._start_segment(timestamp)
        else:
            # Normal continuation, duplicate, or small ordering jitter.
            self._add_to_window(timestamp)

        self._prev_ts = timestamp

    def finalize(self) -> None:
        """EOF: drop any incomplete final window."""
        self._close_partial()

    def _start_segment(self, timestamp: float) -> None:
        self.segment_count += 1
        self._win_count = 1
        self._win_first_ts = timestamp
        self._win_last_ts = timestamp
        if self._win_count == self.policy.window_size:
            self._emit_full_window()

    def _add_to_window(self, timestamp: float) -> None:
        if self._win_count == 0:
            self._win_first_ts = timestamp
            self._win_last_ts = timestamp
            self._win_count = 1
        else:
            self._win_count += 1
            self._win_last_ts = timestamp
        if self._win_count == self.policy.window_size:
            self._emit_full_window()

    def _emit_full_window(self) -> None:
        assert self._win_first_ts is not None and self._win_last_ts is not None
        span = self._win_last_ts - self._win_first_ts
        self.full_window_count += 1
        self.emitted_packet_count += self.policy.window_size
        self.window_spans.append(span)
        if span == 0.0:
            self.zero_span_window_count += 1
        self._win_count = 0
        self._win_first_ts = None
        self._win_last_ts = None

    def _close_partial(self) -> None:
        if self._win_count > 0:
            self.dropped_partial_window_count += 1
            self.dropped_partial_packet_count += self._win_count
            self._win_count = 0
            self._win_first_ts = None
            self._win_last_ts = None

    def to_row(self, meta: dict[str, Any]) -> dict[str, Any]:
        spans = self.window_spans
        retention = (
            self.emitted_packet_count / self.packet_count
            if self.packet_count > 0
            else None
        )
        zero_ratio = (
            self.zero_span_window_count / self.full_window_count
            if self.full_window_count > 0
            else None
        )
        span_sum = sum(spans) if spans else 0.0
        return {
            **meta,
            "window_size": self.policy.window_size,
            "inactivity_timeout_seconds": self.policy.inactivity_timeout_seconds,
            "backward_reset_seconds": self.policy.backward_reset_seconds,
            "packet_count": self.packet_count,
            "positive_gap_boundary_count": self.positive_gap_boundary_count,
            "backward_discontinuity_boundary_count": (
                self.backward_discontinuity_boundary_count
            ),
            "segment_count": self.segment_count,
            "full_window_count": self.full_window_count,
            "emitted_packet_count": self.emitted_packet_count,
            "dropped_partial_window_count": self.dropped_partial_window_count,
            "dropped_partial_packet_count": self.dropped_partial_packet_count,
            "packet_retention_ratio": retention,
            "zero_span_window_count": self.zero_span_window_count,
            "zero_span_window_ratio": zero_ratio,
            "window_span_min": min(spans) if spans else None,
            "window_span_mean": (span_sum / len(spans)) if spans else None,
            "window_span_p50": _percentile(spans, 50),
            "window_span_p95": _percentile(spans, 95),
            "window_span_p99": _percentile(spans, 99),
            "window_span_max": max(spans) if spans else None,
            "windowing_strategy_version": WINDOWING_STRATEGY_VERSION,
        }


def summary_category(meta: dict[str, Any]) -> str | None:
    """Map inventory metadata to a Gate-A summary bucket."""
    source = (meta.get("source") or "").strip()
    label = (meta.get("binary_label") or "").strip()
    family = (meta.get("attack_family") or "").strip()
    profiling_type = (meta.get("profiling_type") or "").strip()

    if source == "attacks" and label == "BENIGN":
        return "publisher_benign"
    if source == "profiling" and profiling_type:
        return f"profiling_{profiling_type}"
    if label == "ATTACK" and family in {
        "DDoS",
        "DoS",
        "MQTT",
        "Recon",
        "Spoofing",
    }:
        return family
    return None


def characterize_timestamps(
    timestamps: list[float] | tuple[float, ...],
    policies: list[WindowPolicy],
) -> list[PolicyAccumulator]:
    """Characterize an in-memory timestamp sequence (tests / small probes)."""
    accs = [PolicyAccumulator(policy=p) for p in policies]
    for ts in timestamps:
        for acc in accs:
            acc.observe(ts)
    for acc in accs:
        acc.finalize()
    return accs


def _default_meta() -> dict[str, Any]:
    return {
        "pcap_path": "",
        "source": "",
        "binary_label": "",
        "attack_family": "",
        "attack_type": "",
        "profiling_type": "",
        "profiling_variant": "",
        "device": "",
        "capture_session": "",
    }


def characterize_pcap(
    pcap_path: Path | str,
    *,
    policies: list[WindowPolicy] | None = None,
    meta: dict[str, Any] | None = None,
    max_packets: int | None = None,
    project_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Scan one PCAP once; update all policies concurrently; return CSV rows."""
    path = Path(pcap_path)
    pols = policies if policies is not None else candidate_policies()
    accs = [PolicyAccumulator(policy=p) for p in pols]

    for _index, ts in iter_timestamps(path, max_packets=max_packets):
        for acc in accs:
            acc.observe(ts)
    for acc in accs:
        acc.finalize()

    row_meta = _default_meta()
    if meta:
        row_meta.update(meta)
    row_meta["pcap_path"] = to_repo_relative(path, project_root=project_root)
    return [acc.to_row(row_meta) for acc in accs]


def _empty(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def load_train_inventory_rows(
    inventory_path: Path | str,
) -> list[dict[str, str]]:
    """Load TRAIN inventory rows (BENIGN/ATTACK only; excludes UNKNOWN)."""
    path = Path(inventory_path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    train: list[dict[str, str]] = []
    for row in rows:
        if row.get("split") != "train":
            continue
        if row.get("binary_label") not in {"BENIGN", "ATTACK"}:
            continue
        train.append(row)
    train.sort(key=lambda r: r["pcap_path"])
    return train


def _meta_from_inventory(row: dict[str, str]) -> dict[str, Any]:
    return {
        "pcap_path": row.get("pcap_path", ""),
        "source": _empty(row.get("source")),
        "binary_label": _empty(row.get("binary_label")),
        "attack_family": _empty(row.get("attack_family")),
        "attack_type": _empty(row.get("attack_type")),
        "profiling_type": _empty(row.get("profiling_type")),
        "profiling_variant": _empty(row.get("profiling_variant")),
        "device": _empty(row.get("device")),
        "capture_session": _empty(row.get("capture_session")),
    }


def _resolve_pcap(
    rel_path: str,
    *,
    project_root: Path,
) -> Path:
    path = Path(rel_path)
    return path if path.is_absolute() else (project_root / path)


def _characterize_one_job(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Picklable worker for process-pool characterization."""
    project_root = Path(payload["project_root"])
    pcap_path = _resolve_pcap(payload["pcap_path"], project_root=project_root)
    policies = [
        WindowPolicy(
            window_size=int(p["window_size"]),
            inactivity_timeout_seconds=float(p["inactivity_timeout_seconds"]),
            backward_reset_seconds=float(p["backward_reset_seconds"]),
        )
        for p in payload["policies"]
    ]
    return characterize_pcap(
        pcap_path,
        policies=policies,
        meta=payload["meta"],
        max_packets=payload.get("max_packets"),
        project_root=project_root,
    )


@dataclass
class CharacterizationResult:
    rows: list[dict[str, Any]]
    output_path: Path
    train_pcap_count: int
    policy_count: int


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = {
                col: ("" if row.get(col) is None else row.get(col)) for col in columns
            }
            writer.writerow(serialized)


def characterize_train_windowing(
    *,
    inventory_path: Path | str | None = None,
    output_path: Path | str | None = None,
    policies: list[WindowPolicy] | None = None,
    project_root: Path | None = None,
    workers: int = DEFAULT_WORKERS,
    max_packets: int | None = None,
    progress_file: TextIO | None = None,
) -> CharacterizationResult:
    """Characterize all TRAIN PCAPs under every candidate windowing policy."""
    root = (project_root or PROJECT_ROOT).resolve()
    inv_path = Path(inventory_path or (DEFAULT_MANIFEST_DIR / "pcap_inventory.csv"))
    if not inv_path.is_absolute():
        inv_path = root / inv_path
    out = Path(output_path or DEFAULT_CHARACTERIZATION_CSV)
    if not out.is_absolute():
        out = root / out

    pols = policies if policies is not None else candidate_policies()
    train_rows = load_train_inventory_rows(inv_path)
    progress = progress_file

    policy_payload = [
        {
            "window_size": p.window_size,
            "inactivity_timeout_seconds": p.inactivity_timeout_seconds,
            "backward_reset_seconds": p.backward_reset_seconds,
        }
        for p in pols
    ]
    jobs = [
        {
            "project_root": str(root),
            "pcap_path": row["pcap_path"],
            "meta": _meta_from_inventory(row),
            "policies": policy_payload,
            "max_packets": max_packets,
        }
        for row in train_rows
    ]

    all_rows: list[dict[str, Any]] = []
    worker_count = max(1, int(workers))

    if progress is not None:
        print(
            f"Phase 1C.1 windowing characterization: "
            f"{len(jobs)} TRAIN PCAPs × {len(pols)} policies "
            f"(workers={worker_count})",
            file=progress,
            flush=True,
        )

    if worker_count == 1:
        for i, job in enumerate(jobs, start=1):
            if progress is not None:
                print(
                    f"[{i}/{len(jobs)}] {job['pcap_path']}",
                    file=progress,
                    flush=True,
                )
            all_rows.extend(_characterize_one_job(job))
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_characterize_one_job, job): job for job in jobs
            }
            done = 0
            for done, future in enumerate(as_completed(futures), start=1):
                job = futures[future]
                rows = future.result()
                all_rows.extend(rows)
                if progress is not None:
                    print(
                        f"[{done}/{len(jobs)}] {job['pcap_path']}",
                        file=progress,
                        flush=True,
                    )

    all_rows.sort(
        key=lambda r: (
            r.get("pcap_path", ""),
            int(r.get("window_size", 0)),
            float(r.get("inactivity_timeout_seconds", 0.0)),
        )
    )
    _write_csv(out, all_rows, CHARACTERIZATION_COLUMNS)
    return CharacterizationResult(
        rows=all_rows,
        output_path=out,
        train_pcap_count=len(train_rows),
        policy_count=len(pols),
    )


def format_characterization_summary(rows: list[dict[str, Any]]) -> str:
    """Aggregate Gate-A summary by category × policy (TRAIN only)."""
    policies: dict[tuple[int, float, float], None] = {}
    buckets: dict[tuple[int, float, float], dict[str, dict[str, Any]]] = {}

    for row in rows:
        key = (
            int(row["window_size"]),
            float(row["inactivity_timeout_seconds"]),
            float(row["backward_reset_seconds"]),
        )
        policies[key] = None
        cat = summary_category(row)
        if cat is None:
            continue
        by_cat = buckets.setdefault(key, {})
        agg = by_cat.setdefault(
            cat,
            {
                "pcap_count": 0,
                "packet_count": 0,
                "full_window_count": 0,
                "emitted_packet_count": 0,
                "dropped_partial_packet_count": 0,
                "segment_count": 0,
                "zero_span_window_count": 0,
                "window_span_p50_list": [],
                "window_span_p95_list": [],
                "window_span_p99_list": [],
            },
        )
        agg["pcap_count"] += 1
        agg["packet_count"] += int(row["packet_count"] or 0)
        agg["full_window_count"] += int(row["full_window_count"] or 0)
        agg["emitted_packet_count"] += int(row["emitted_packet_count"] or 0)
        agg["dropped_partial_packet_count"] += int(
            row["dropped_partial_packet_count"] or 0
        )
        agg["segment_count"] += int(row["segment_count"] or 0)
        agg["zero_span_window_count"] += int(row["zero_span_window_count"] or 0)
        for span_key in (
            "window_span_p50",
            "window_span_p95",
            "window_span_p99",
        ):
            val = row.get(span_key)
            if val is not None and val != "":
                agg[f"{span_key}_list"].append(float(val))

    lines: list[str] = [
        "Phase 1C.1 Windowing Characterization Summary (TRAIN only)",
        "=" * 64,
        "",
        "GATE A — STOP FOR REVIEW",
        "Do not choose WINDOW_SIZE automatically.",
        "Do not scan TEST.",
        "Do not generate the full feature dataset.",
        "",
    ]

    for window_size, inactivity, backward in sorted(policies.keys()):
        lines.append(
            f"## config window_size={window_size}  "
            f"inactivity={inactivity:g}s  backward_reset={backward:g}s"
        )
        lines.append("-" * 64)
        header = (
            f"{'category':<24} {'pcaps':>5} {'full_win':>10} {'retain':>8} "
            f"{'tail_loss':>9} {'segs':>8} {'zero_r':>7} "
            f"{'span_p50':>10} {'span_p95':>10} {'span_p99':>10}"
        )
        lines.append(header)
        by_cat = buckets.get((window_size, inactivity, backward), {})
        for cat in SUMMARY_CATEGORY_ORDER:
            agg = by_cat.get(cat)
            if agg is None:
                continue
            packets = agg["packet_count"]
            retain = (
                agg["emitted_packet_count"] / packets if packets > 0 else float("nan")
            )
            tail_loss = (
                agg["dropped_partial_packet_count"] / packets
                if packets > 0
                else float("nan")
            )
            zero_r = (
                agg["zero_span_window_count"] / agg["full_window_count"]
                if agg["full_window_count"] > 0
                else float("nan")
            )

            def _mean_span(name: str, _agg: dict[str, Any] = agg) -> str:
                vals = _agg.get(f"{name}_list", [])
                if not vals:
                    return "n/a"
                return f"{sum(vals) / len(vals):.4g}"

            lines.append(
                f"{cat:<24} {int(agg['pcap_count']):>5} "
                f"{int(agg['full_window_count']):>10} {retain:>8.3f} "
                f"{tail_loss:>9.3f} {int(agg['segment_count']):>8} "
                f"{zero_r:>7.3f} "
                f"{_mean_span('window_span_p50'):>10} "
                f"{_mean_span('window_span_p95'):>10} "
                f"{_mean_span('window_span_p99'):>10}"
            )
        lines.append("")

    lines.extend(
        [
            "Notes:",
            "- retain = emitted_packets / packets (packet retention)",
            "- tail_loss = dropped_partial_packets / packets",
            "- zero_r = zero-span windows / full windows",
            (
                "- span_p50/p95/p99 are means of per-PCAP percentiles "
                "(category-level approximation)"
            ),
            "- Pay particular attention to publisher_benign and profiling_* retention",
            "",
            "Freeze after review:",
            "  WINDOW_SIZE = ?",
            "  INACTIVITY_TIMEOUT_SECONDS = ?",
            f"  BACKWARD_RESET_SECONDS = {DEFAULT_BACKWARD_RESET_SECONDS}",
        ]
    )
    return "\n".join(lines)
