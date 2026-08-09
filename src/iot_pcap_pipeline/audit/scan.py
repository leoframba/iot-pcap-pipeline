"""Corpus audit scan orchestration with parallel workers and resume."""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from iot_pcap_pipeline.audit.checkpoints import CheckpointStore
from iot_pcap_pipeline.audit.issues import IssueCollector
from iot_pcap_pipeline.audit.live_progress import (
    DEFAULT_PROGRESS_EVERY_PACKETS,
    LiveProgressReporter,
    LiveProgressStore,
)
from iot_pcap_pipeline.audit.policy import (
    DEFAULT_ISSUE_CAP_PER_CODE,
    DEFAULT_MALFORMED_CATASTROPHIC_RATE,
    DEFAULT_MALFORMED_HIGH_WARNING_RATE,
    DEFAULT_WORKERS,
)
from iot_pcap_pipeline.audit.reconcile import reconcile_manifests
from iot_pcap_pipeline.audit.schema import (
    AUDIT_ISSUE_COLUMNS,
    INTEGRITY_COLUMNS,
    TRAIN_CHARACTERIZATION_COLUMNS,
)
from iot_pcap_pipeline.audit.summary import format_audit_summary
from iot_pcap_pipeline.audit.worker import (
    AuditPolicy,
    PcapAuditResult,
    scan_one_pcap_result,
    worker_crash_result,
)
from iot_pcap_pipeline.paths import (
    DEFAULT_AUDIT_DIR,
    DEFAULT_MANIFEST_DIR,
    DEFAULT_RAW_ROOT,
    PROJECT_ROOT,
)
from iot_pcap_pipeline.pcap.stats import DEFAULT_IP_CARDINALITY_CAP


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


def _format_bytes(num: float) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num} B"


@dataclass
class AuditResult:
    integrity_rows: list[dict[str, Any]] = field(default_factory=list)
    train_rows: list[dict[str, Any]] = field(default_factory=list)
    issue_rows: list[dict[str, Any]] = field(default_factory=list)
    hard_fail: bool = False
    summary: str = ""
    integrity_path: Path | None = None
    train_path: Path | None = None
    issues_path: Path | None = None
    checkpoint_hits: int = 0
    scanned_files: int = 0
    total_scan_seconds: float | None = None


def _sort_issue_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            r.get("scope") or "",
            r.get("pcap_path") or "",
            r.get("packet_index") if r.get("packet_index") is not None else -1,
            r.get("issue_code") or "",
        ),
    )


def _report_progress(
    progress_file: TextIO | None,
    *,
    completed: int,
    total: int,
    result: PcapAuditResult,
    bytes_done: int,
    bytes_total: int,
    cached: int,
    workers: int,
) -> None:
    if progress_file is None:
        return
    name = Path(result.pcap_path).name
    packets = result.packet_count
    elapsed = result.scan_elapsed_seconds
    source = "checkpoint" if result.from_checkpoint else "scan"
    pkt_txt = f"{packets:,}" if packets is not None else "n/a"
    elapsed_txt = f"{elapsed:.1f}s" if elapsed is not None else "n/a"
    progress_file.write(
        f"[{completed}/{total} complete] {name} ({source})\n"
        f"  packets: {pkt_txt}\n"
        f"  elapsed: {elapsed_txt}\n"
        f"Processed: {_format_bytes(bytes_done)} / {_format_bytes(bytes_total)} "
        f"({(100.0 * bytes_done / bytes_total) if bytes_total else 0.0:.1f}%)\n"
        f"Files:     {completed} / {total}\n"
        f"Cached:    {cached}\n"
        f"Workers:   {workers}\n"
    )
    progress_file.flush()


def audit_corpus(
    *,
    inventory_path: Path | None = None,
    split_path: Path | None = None,
    raw_root: Path | None = None,
    output_dir: Path | None = None,
    project_root: Path | None = None,
    ip_cardinality_cap: int = DEFAULT_IP_CARDINALITY_CAP,
    issue_cap_per_code: int = DEFAULT_ISSUE_CAP_PER_CODE,
    malformed_high_rate: float = DEFAULT_MALFORMED_HIGH_WARNING_RATE,
    malformed_catastrophic_rate: float = DEFAULT_MALFORMED_CATASTROPHIC_RATE,
    workers: int = DEFAULT_WORKERS,
    resume: bool = True,
    checkpoint_dir: Path | None = None,
    clear_checkpoints: bool = False,
    progress_every_packets: int = DEFAULT_PROGRESS_EVERY_PACKETS,
    progress_file: TextIO | None = None,
    summary_file: TextIO | None = None,
) -> AuditResult:
    root = (project_root or PROJECT_ROOT).resolve()
    manifest_dir = DEFAULT_MANIFEST_DIR
    inv_path = (inventory_path or (manifest_dir / "pcap_inventory.csv")).resolve()
    spl_path = (split_path or (manifest_dir / "dataset_split.csv")).resolve()
    raw = (raw_root or DEFAULT_RAW_ROOT).resolve()
    out = (output_dir or DEFAULT_AUDIT_DIR).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = (
        checkpoint_dir if checkpoint_dir is not None else (out / ".work")
    ).resolve()
    store = CheckpointStore(ckpt_dir)
    live_store = LiveProgressStore(ckpt_dir / "in_progress")
    if clear_checkpoints:
        store.clear()
        live_store.clear_all()
    else:
        live_store.clear_all()

    policy = AuditPolicy(
        ip_cardinality_cap=ip_cardinality_cap,
        issue_cap_per_code=issue_cap_per_code,
        malformed_high_rate=malformed_high_rate,
        malformed_catastrophic_rate=malformed_catastrophic_rate,
        progress_dir=str(live_store.root),
        progress_every_packets=progress_every_packets,
    )

    corpus_issues = IssueCollector(issue_cap_per_code=issue_cap_per_code)
    reconciliation = reconcile_manifests(
        inventory_path=inv_path,
        split_path=spl_path,
        raw_root=raw,
        project_root=root,
        issues=corpus_issues,
    )

    # Schedule largest first for better worker balance; final outputs still sorted.
    entries = list(reconciliation.inventory_rows)
    entries.sort(
        key=lambda row: int(row.get("file_size") or 0),
        reverse=True,
    )
    bytes_total = sum(int(row.get("file_size") or 0) for row in entries)
    total = len(entries)

    results_by_path: dict[str, PcapAuditResult] = {}
    to_scan: list[dict[str, Any]] = []
    checkpoint_hits = 0
    bytes_done = 0

    for row in entries:
        pcap_path = row["pcap_path"]
        abs_path = root / pcap_path
        disk_size = abs_path.stat().st_size if abs_path.is_file() else None
        cached = None
        if resume:
            cached = store.load_valid(
                pcap_path=pcap_path,
                policy=policy,
                manifest_file_size=row.get("file_size"),
                disk_file_size=disk_size,
                split=row.get("split"),
            )
        if cached is not None:
            results_by_path[pcap_path] = cached
            checkpoint_hits += 1
            bytes_done += int(cached.file_size or row.get("file_size") or 0)
            _report_progress(
                progress_file,
                completed=len(results_by_path),
                total=total,
                result=cached,
                bytes_done=bytes_done,
                bytes_total=bytes_total,
                cached=checkpoint_hits,
                workers=workers,
            )
        else:
            to_scan.append(row)

    scanned_files = 0
    total_scan_seconds = 0.0
    live_reporter = LiveProgressReporter(live_store, progress_file)

    def _ingest(result: PcapAuditResult, *, checkpoint: bool) -> None:
        nonlocal bytes_done, scanned_files, total_scan_seconds
        live_store.clear(result.pcap_path)
        results_by_path[result.pcap_path] = result
        if (
            checkpoint
            and not result.from_checkpoint
            and "file:worker_crash" not in result.hard_failures
        ):
            store.save(result, policy=policy)
        if not result.from_checkpoint:
            scanned_files += 1
            if result.scan_elapsed_seconds is not None:
                total_scan_seconds += result.scan_elapsed_seconds
        bytes_done += int(result.file_size or result.integrity_row.get("file_size_manifest") or 0)
        _report_progress(
            progress_file,
            completed=len(results_by_path),
            total=total,
            result=result,
            bytes_done=bytes_done,
            bytes_total=bytes_total,
            cached=checkpoint_hits,
            workers=workers,
        )

    if to_scan:
        root_str = str(root)
        worker_count = max(1, int(workers))
        if progress_file is not None:
            progress_file.write(
                f"Scanning {len(to_scan)} PCAPs "
                f"({_format_bytes(sum(int(r.get('file_size') or 0) for r in to_scan))}) "
                f"with {worker_count} worker(s); "
                f"live packet updates every {progress_every_packets:,} packets\n"
            )
            for row in to_scan[:worker_count]:
                progress_file.write(f"  → starting {Path(row['pcap_path']).name}\n")
            progress_file.flush()
        live_reporter.start()
        try:
            if worker_count == 1:
                for row in to_scan:
                    result = scan_one_pcap_result(row, root_str, policy)
                    _ingest(result, checkpoint=True)
            else:
                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    future_map = {
                        executor.submit(
                            scan_one_pcap_result, row, root_str, policy
                        ): row
                        for row in to_scan
                    }
                    for future in as_completed(future_map):
                        row = future_map[future]
                        try:
                            result = future.result()
                        except Exception as exc:  # noqa: BLE001 - isolate worker crashes
                            result = worker_crash_result(
                                row,
                                detail=f"worker crashed: {type(exc).__name__}: {exc}",
                            )
                        _ingest(result, checkpoint=True)
        finally:
            live_reporter.stop()
            live_store.clear_all()

    # Deterministic merge regardless of completion order.
    ordered_paths = sorted(results_by_path.keys())
    integrity_rows = [results_by_path[p].integrity_row for p in ordered_paths]
    train_rows = [
        results_by_path[p].training_row
        for p in ordered_paths
        if results_by_path[p].training_row is not None
    ]
    # train_rows already follows path order because ordered_paths is sorted.

    issue_rows = list(corpus_issues.rows())
    for path in ordered_paths:
        issue_rows.extend(results_by_path[path].issue_rows)
    issue_rows = _sort_issue_rows(issue_rows)

    by_protocol_by_path = {
        path: results_by_path[path].by_protocol for path in ordered_paths
    }

    integrity_path = out / "pcap_integrity.csv"
    train_path = out / "training_characterization.csv"
    issues_path = out / "audit_issues.csv"
    _write_csv(integrity_path, integrity_rows, INTEGRITY_COLUMNS)
    _write_csv(train_path, train_rows, TRAIN_CHARACTERIZATION_COLUMNS)
    _write_csv(issues_path, issue_rows, AUDIT_ISSUE_COLUMNS)

    hard_fail = reconciliation.hard_fail or any(
        results_by_path[p].hard_failures for p in ordered_paths
    ) or corpus_issues.has_hard_failures()

    # Runtime metadata for summary.
    slowest = sorted(
        (
            results_by_path[p]
            for p in ordered_paths
            if results_by_path[p].scan_elapsed_seconds is not None
        ),
        key=lambda r: r.scan_elapsed_seconds or 0.0,
        reverse=True,
    )[:5]

    summary = format_audit_summary(
        integrity_rows=integrity_rows,
        train_rows=train_rows,
        issue_rows=issue_rows,
        by_protocol_by_path=by_protocol_by_path,
        hard_fail=hard_fail,
        checkpoint_hits=checkpoint_hits,
        scanned_files=scanned_files,
        total_scan_seconds=total_scan_seconds if scanned_files else None,
        slowest=slowest,
        workers=workers,
    )
    if summary_file is not None:
        summary_file.write(summary)
    else:
        print(summary, end="")

    return AuditResult(
        integrity_rows=integrity_rows,
        train_rows=train_rows,
        issue_rows=issue_rows,
        hard_fail=hard_fail,
        summary=summary,
        integrity_path=integrity_path,
        train_path=train_path,
        issues_path=issues_path,
        checkpoint_hits=checkpoint_hits,
        scanned_files=scanned_files,
        total_scan_seconds=total_scan_seconds if scanned_files else None,
    )
