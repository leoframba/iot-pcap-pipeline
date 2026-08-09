"""Live per-PCAP packet progress for parallel audits."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

DEFAULT_PROGRESS_EVERY_PACKETS = 250_000


def _progress_id(pcap_path: str) -> str:
    return hashlib.sha256(pcap_path.encode("utf-8")).hexdigest()


@dataclass
class LiveProgress:
    pcap_path: str
    filename: str
    packets: int
    elapsed_seconds: float
    status: str  # starting | running | done
    file_size: int | None = None


class LiveProgressStore:
    """Worker-written progress sidecars under checkpoint_dir/in_progress/."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, pcap_path: str) -> Path:
        return self.root / f"{_progress_id(pcap_path)}.progress.json"

    def tmp_path_for(self, pcap_path: str) -> Path:
        return self.root / f"{_progress_id(pcap_path)}.progress.json.tmp"

    def write(
        self,
        *,
        pcap_path: str,
        packets: int,
        elapsed_seconds: float,
        status: str,
        file_size: int | None = None,
    ) -> None:
        payload = {
            "pcap_path": pcap_path,
            "filename": Path(pcap_path).name,
            "packets": int(packets),
            "elapsed_seconds": float(elapsed_seconds),
            "status": status,
            "file_size": file_size,
            "updated_at": time.time(),
        }
        path = self.path_for(pcap_path)
        tmp = self.tmp_path_for(pcap_path)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    def clear(self, pcap_path: str) -> None:
        self.path_for(pcap_path).unlink(missing_ok=True)
        self.tmp_path_for(pcap_path).unlink(missing_ok=True)

    def clear_all(self) -> None:
        for path in self.root.glob("*.progress.json"):
            path.unlink(missing_ok=True)
        for path in self.root.glob("*.progress.json.tmp"):
            path.unlink(missing_ok=True)

    def read_all(self) -> list[LiveProgress]:
        items: list[LiveProgress] = []
        for path in sorted(self.root.glob("*.progress.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            items.append(
                LiveProgress(
                    pcap_path=str(payload.get("pcap_path") or ""),
                    filename=str(payload.get("filename") or ""),
                    packets=int(payload.get("packets") or 0),
                    elapsed_seconds=float(payload.get("elapsed_seconds") or 0.0),
                    status=str(payload.get("status") or "running"),
                    file_size=(
                        int(payload["file_size"])
                        if payload.get("file_size") is not None
                        else None
                    ),
                )
            )
        return items


class LiveProgressReporter:
    """Parent-side poller that prints active worker packet progress."""

    def __init__(
        self,
        store: LiveProgressStore,
        progress_file: TextIO | None,
        *,
        poll_seconds: float = 2.0,
    ) -> None:
        self.store = store
        self.progress_file = progress_file
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_line: dict[str, str] = {}

    def start(self) -> None:
        if self.progress_file is None:
            return
        self._thread = threading.Thread(
            target=self._run, name="audit-live-progress", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self) -> None:
        assert self.progress_file is not None
        while not self._stop.wait(self.poll_seconds):
            actives = [
                item
                for item in self.store.read_all()
                if item.status in {"starting", "running"}
            ]
            if not actives:
                continue
            for item in sorted(actives, key=lambda x: x.filename):
                pps = (
                    item.packets / item.elapsed_seconds
                    if item.elapsed_seconds > 0
                    else 0.0
                )
                line = (
                    f"  … {item.filename}: {item.packets:,} packets "
                    f"({item.elapsed_seconds:.0f}s, {pps:,.0f} pkt/s)"
                )
                if self._last_line.get(item.pcap_path) == line:
                    continue
                self._last_line[item.pcap_path] = line
                self.progress_file.write(line + "\n")
            self.progress_file.flush()
