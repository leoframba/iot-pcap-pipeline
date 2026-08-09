"""Phase 1B.3 timestamp-only ordering probe.

Streams PCAP timestamps in original capture order via dpkt without decoding
frames. Computes adjacent-delta statistics to characterize negative timestamp
deltas before Phase 1C temporal feature design.

Positive-delta percentiles use a deterministic reservoir sample when the
positive count exceeds ``positive_sample_cap``; negative-delta percentiles are
always exact. Min / max / mean for both classes are exact streaming values.
"""

from __future__ import annotations

import csv
import heapq
import math
import random
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import dpkt

from iot_pcap_pipeline.paths import (
    DEFAULT_AUDIT_DIR,
    PROJECT_ROOT,
    TIMESTAMP_PROBE_STRATEGY_VERSION,
    to_repo_relative,
)

DEFAULT_EXAMPLE_LIMIT = 10
DEFAULT_LARGEST_EXAMPLE_LIMIT = 10
DEFAULT_POSITIVE_SAMPLE_CAP = 100_000
DEFAULT_POSITIVE_SAMPLE_SEED = 42

DEFAULT_PROBE_CSV = DEFAULT_AUDIT_DIR / "timestamp_probe.csv"
DEFAULT_EXAMPLES_CSV = DEFAULT_AUDIT_DIR / "timestamp_reversal_examples.csv"

ExampleKind = Literal["first", "largest"]

PROBE_COLUMNS: list[str] = [
    "pcap_path",
    "packet_count",
    "adjacent_delta_count",
    "positive_delta_count",
    "duplicate_delta_count",
    "negative_delta_count",
    "duplicate_ratio",
    "negative_delta_ratio",
    "negative_delta_min_magnitude",
    "negative_delta_max_magnitude",
    "negative_delta_mean_magnitude",
    "negative_delta_p50_magnitude",
    "negative_delta_p95_magnitude",
    "negative_delta_p99_magnitude",
    "positive_delta_min",
    "positive_delta_max",
    "positive_delta_mean",
    "positive_delta_p50",
    "positive_delta_p95",
    "positive_delta_p99",
    "positive_percentile_method",
    "positive_percentile_sample_size",
    "negative_run_count",
    "negative_run_max_length",
    "negative_run_mean_length",
    "probe_strategy_version",
]

EXAMPLE_COLUMNS: list[str] = [
    "pcap_path",
    "example_kind",
    "packet_index_previous",
    "packet_index_current",
    "previous_timestamp",
    "current_timestamp",
    "delta_seconds",
    "delta_magnitude_seconds",
]


@dataclass(frozen=True)
class ReversalExample:
    """Bounded sample of a negative adjacent timestamp delta."""

    pcap_path: str
    packet_index_previous: int
    packet_index_current: int
    previous_timestamp: float
    current_timestamp: float
    delta_seconds: float
    delta_magnitude_seconds: float
    example_kind: ExampleKind = "first"


@dataclass
class TimestampProbeResult:
    """Per-PCAP timestamp ordering probe summary."""

    pcap_path: str
    packet_count: int = 0
    adjacent_delta_count: int = 0
    positive_delta_count: int = 0
    duplicate_delta_count: int = 0
    negative_delta_count: int = 0
    duplicate_ratio: float | None = None
    negative_delta_ratio: float | None = None
    negative_delta_min_magnitude: float | None = None
    negative_delta_max_magnitude: float | None = None
    negative_delta_mean_magnitude: float | None = None
    negative_delta_p50_magnitude: float | None = None
    negative_delta_p95_magnitude: float | None = None
    negative_delta_p99_magnitude: float | None = None
    positive_delta_min: float | None = None
    positive_delta_max: float | None = None
    positive_delta_mean: float | None = None
    positive_delta_p50: float | None = None
    positive_delta_p95: float | None = None
    positive_delta_p99: float | None = None
    positive_percentile_method: str = "exact"
    positive_percentile_sample_size: int = 0
    negative_run_count: int = 0
    negative_run_max_length: int = 0
    negative_run_mean_length: float | None = None
    probe_strategy_version: str = TIMESTAMP_PROBE_STRATEGY_VERSION
    examples: list[ReversalExample] = field(default_factory=list)
    largest_examples: list[ReversalExample] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        """Serialize summary fields for CSV (excludes examples)."""
        row = asdict(self)
        row.pop("examples", None)
        row.pop("largest_examples", None)
        return row

    def all_examples(self) -> list[ReversalExample]:
        """First-N examples followed by largest-N examples (may overlap)."""
        return [*self.examples, *self.largest_examples]


class _ReservoirSampler:
    """Deterministic reservoir sample for streaming positive deltas."""

    def __init__(self, capacity: int, seed: int) -> None:
        if capacity < 1:
            raise ValueError("positive_sample_cap must be >= 1")
        self.capacity = capacity
        self._rng = random.Random(seed)
        self.samples: list[float] = []
        self.seen = 0

    def add(self, value: float) -> None:
        self.seen += 1
        if len(self.samples) < self.capacity:
            self.samples.append(value)
            return
        j = self._rng.randint(0, self.seen - 1)
        if j < self.capacity:
            self.samples[j] = value


def _percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolation percentile over an unsorted list (p in 0..100)."""
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


def iter_timestamps(
    pcap_path: Path | str,
    *,
    max_packets: int | None = None,
) -> Iterator[tuple[int, float]]:
    """Yield ``(packet_index, timestamp)`` in original capture order.

    Opens the PCAP read-only. Frame buffers are ignored entirely — no
    Ethernet/IP/TCP decoding is performed.
    """
    path = Path(pcap_path)
    if not path.is_file():
        raise FileNotFoundError(f"PCAP not found: {path}")

    with path.open("rb") as handle:
        try:
            reader = dpkt.pcap.Reader(handle)
        except (ValueError, OSError, dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError) as exc:
            raise ValueError(f"failed to open PCAP {path}: {exc}") from exc

        for index, (timestamp, _buf) in enumerate(reader):
            if max_packets is not None and index >= max_packets:
                break
            yield index, float(timestamp)


def probe_timestamps(
    pcap_path: Path | str,
    *,
    example_limit: int = DEFAULT_EXAMPLE_LIMIT,
    largest_example_limit: int = DEFAULT_LARGEST_EXAMPLE_LIMIT,
    positive_sample_cap: int = DEFAULT_POSITIVE_SAMPLE_CAP,
    positive_sample_seed: int = DEFAULT_POSITIVE_SAMPLE_SEED,
    max_packets: int | None = None,
    project_root: Path | None = None,
) -> TimestampProbeResult:
    """Probe adjacent timestamp ordering for a single PCAP.

    Classification of ``delta = ts_i - ts_(i-1)``:

    - ``delta > 0`` → positive
    - ``delta = 0`` → duplicate
    - ``delta < 0`` → negative / reversal

    Values are never modified. Packets are never sorted.

    Retains both the first ``example_limit`` reversals (for burst locality) and
    the ``largest_example_limit`` reversals by magnitude (for outlier location).
    """
    path = Path(pcap_path)
    rel = to_repo_relative(path, project_root=project_root)
    result = TimestampProbeResult(pcap_path=rel)

    if example_limit < 0:
        raise ValueError("example_limit must be >= 0")
    if largest_example_limit < 0:
        raise ValueError("largest_example_limit must be >= 0")

    prev_ts: float | None = None
    prev_index: int | None = None

    negative_magnitudes: list[float] = []
    positive_sum = 0.0
    positive_min: float | None = None
    positive_max: float | None = None
    reservoir = _ReservoirSampler(positive_sample_cap, positive_sample_seed)

    current_run_length = 0
    run_lengths: list[int] = []
    # Min-heap of (magnitude, packet_index_current, example) → keep largest N.
    largest_heap: list[tuple[float, int, ReversalExample]] = []

    for index, ts in iter_timestamps(path, max_packets=max_packets):
        result.packet_count += 1

        if prev_ts is None or prev_index is None:
            prev_ts = ts
            prev_index = index
            continue

        delta = ts - prev_ts
        result.adjacent_delta_count += 1

        if delta > 0:
            result.positive_delta_count += 1
            positive_sum += delta
            positive_min = delta if positive_min is None else min(positive_min, delta)
            positive_max = delta if positive_max is None else max(positive_max, delta)
            reservoir.add(delta)
            if current_run_length > 0:
                run_lengths.append(current_run_length)
                current_run_length = 0
        elif delta == 0:
            result.duplicate_delta_count += 1
            if current_run_length > 0:
                run_lengths.append(current_run_length)
                current_run_length = 0
        else:
            magnitude = -delta
            result.negative_delta_count += 1
            negative_magnitudes.append(magnitude)
            current_run_length += 1
            example = ReversalExample(
                pcap_path=rel,
                packet_index_previous=prev_index,
                packet_index_current=index,
                previous_timestamp=prev_ts,
                current_timestamp=ts,
                delta_seconds=delta,
                delta_magnitude_seconds=magnitude,
                example_kind="first",
            )
            if len(result.examples) < example_limit:
                result.examples.append(example)
            if largest_example_limit > 0:
                largest = ReversalExample(
                    pcap_path=rel,
                    packet_index_previous=prev_index,
                    packet_index_current=index,
                    previous_timestamp=prev_ts,
                    current_timestamp=ts,
                    delta_seconds=delta,
                    delta_magnitude_seconds=magnitude,
                    example_kind="largest",
                )
                item = (magnitude, index, largest)
                if len(largest_heap) < largest_example_limit:
                    heapq.heappush(largest_heap, item)
                elif magnitude > largest_heap[0][0]:
                    heapq.heapreplace(largest_heap, item)

        prev_ts = ts
        prev_index = index

    if current_run_length > 0:
        run_lengths.append(current_run_length)

    if largest_heap:
        result.largest_examples = [
            ex
            for _mag, _idx, ex in sorted(
                largest_heap, key=lambda item: (-item[0], item[1])
            )
        ]

    n_adj = result.adjacent_delta_count
    if n_adj > 0:
        result.duplicate_ratio = result.duplicate_delta_count / n_adj
        result.negative_delta_ratio = result.negative_delta_count / n_adj

    if negative_magnitudes:
        neg_sum = sum(negative_magnitudes)
        result.negative_delta_min_magnitude = min(negative_magnitudes)
        result.negative_delta_max_magnitude = max(negative_magnitudes)
        result.negative_delta_mean_magnitude = neg_sum / len(negative_magnitudes)
        result.negative_delta_p50_magnitude = _percentile(negative_magnitudes, 50)
        result.negative_delta_p95_magnitude = _percentile(negative_magnitudes, 95)
        result.negative_delta_p99_magnitude = _percentile(negative_magnitudes, 99)

    if result.positive_delta_count > 0:
        result.positive_delta_min = positive_min
        result.positive_delta_max = positive_max
        result.positive_delta_mean = positive_sum / result.positive_delta_count
        result.positive_percentile_sample_size = len(reservoir.samples)
        if result.positive_delta_count <= positive_sample_cap:
            result.positive_percentile_method = "exact"
        else:
            result.positive_percentile_method = (
                f"reservoir_n{positive_sample_cap}_seed{positive_sample_seed}"
            )
        result.positive_delta_p50 = _percentile(reservoir.samples, 50)
        result.positive_delta_p95 = _percentile(reservoir.samples, 95)
        result.positive_delta_p99 = _percentile(reservoir.samples, 99)

    result.negative_run_count = len(run_lengths)
    if run_lengths:
        result.negative_run_max_length = max(run_lengths)
        result.negative_run_mean_length = sum(run_lengths) / len(run_lengths)

    return result


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


def write_probe_artifacts(
    results: list[TimestampProbeResult],
    *,
    output_path: Path | str = DEFAULT_PROBE_CSV,
    examples_path: Path | str | None = DEFAULT_EXAMPLES_CSV,
) -> dict[str, Path]:
    """Write summary and optional reversal-example CSVs."""
    out = Path(output_path)
    _write_csv(out, [r.to_row() for r in results], PROBE_COLUMNS)

    written: dict[str, Path] = {"probe_path": out}
    if examples_path is not None:
        ex_path = Path(examples_path)
        example_rows = [asdict(ex) for r in results for ex in r.all_examples()]
        _write_csv(ex_path, example_rows, EXAMPLE_COLUMNS)
        written["examples_path"] = ex_path
    return written


def format_probe_summary(result: TimestampProbeResult) -> str:
    """Human-readable CLI summary emphasizing negative-delta magnitudes."""

    def _us(seconds: float | None) -> str:
        if seconds is None:
            return "n/a"
        return f"{seconds * 1e6:.3f} µs ({seconds:.9f} s)"

    def _format_example(ex: ReversalExample) -> str:
        return (
            f"  idx {ex.packet_index_previous}->{ex.packet_index_current}: "
            f"Δ={ex.delta_seconds:.9f} s  "
            f"|Δ|={ex.delta_magnitude_seconds * 1e6:.3f} µs "
            f"({ex.delta_magnitude_seconds:.9f} s)"
        )

    lines = [
        f"\n=== {result.pcap_path} ===",
        f"packets: {result.packet_count:,}",
        f"adjacent_deltas: {result.adjacent_delta_count:,}",
        (
            f"positive/duplicate/negative: "
            f"{result.positive_delta_count:,} / "
            f"{result.duplicate_delta_count:,} / "
            f"{result.negative_delta_count:,}"
        ),
    ]
    if result.negative_delta_ratio is not None:
        lines.append(
            f"negative_ratio: {result.negative_delta_ratio:.6e}  "
            f"duplicate_ratio: {result.duplicate_ratio:.6e}"
        )
    lines.append(
        "negative magnitude  "
        f"p50={_us(result.negative_delta_p50_magnitude)}  "
        f"p95={_us(result.negative_delta_p95_magnitude)}  "
        f"p99={_us(result.negative_delta_p99_magnitude)}  "
        f"max={_us(result.negative_delta_max_magnitude)}"
    )
    lines.append(
        "positive delta      "
        f"p50={_us(result.positive_delta_p50)}  "
        f"p95={_us(result.positive_delta_p95)}  "
        f"p99={_us(result.positive_delta_p99)}  "
        f"max={_us(result.positive_delta_max)}"
    )
    lines.append(
        f"positive_percentile_method: {result.positive_percentile_method} "
        f"(sample_size={result.positive_percentile_sample_size})"
    )
    if result.negative_run_count:
        mean_run = (
            f"{result.negative_run_mean_length:.2f}"
            if result.negative_run_mean_length is not None
            else "n/a"
        )
        lines.append(
            f"negative_runs: count={result.negative_run_count}  "
            f"max_len={result.negative_run_max_length}  mean_len={mean_run}"
        )
    else:
        lines.append("negative_runs: none")
    if result.largest_examples:
        lines.append(
            f"largest_reversals (top {len(result.largest_examples)} by magnitude):"
        )
        for ex in result.largest_examples:
            lines.append(_format_example(ex))
    if result.examples:
        lines.append(f"first_reversals (first {len(result.examples)}):")
        for ex in result.examples:
            lines.append(_format_example(ex))
    return "\n".join(lines)


def resolve_pcap_path(raw: Path, *, project_root: Path | None = None) -> Path:
    """Resolve a CLI path relative to the project root when needed."""
    root = project_root or PROJECT_ROOT
    return raw if raw.is_absolute() else (root / raw)
