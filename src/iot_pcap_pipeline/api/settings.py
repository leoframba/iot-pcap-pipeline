"""Runtime configuration for the HTTP serving surface."""

from __future__ import annotations

import os
from dataclasses import dataclass

# Provisional default only — D4 benchmarking chooses a production limit.
DEFAULT_MAX_PCAP_BYTES = 512 * 1024 * 1024
DEFAULT_INPUT_BUCKET = "iomt-input"
DEFAULT_INPUT_PREFIX = "pcaps/"
DEFAULT_PCAP_FETCHER = "gcs"


@dataclass(frozen=True)
class ServingSettings:
    """Application-level GCS allowlist + download limits."""

    input_bucket: str
    input_prefix: str
    max_pcap_bytes: int
    # "gcs" (production) or "local" (Docker/CI smoke without GCP credentials).
    pcap_fetcher: str = DEFAULT_PCAP_FETCHER
    local_pcap_root: str | None = None

    def __post_init__(self) -> None:
        bucket = self.input_bucket.strip()
        if not bucket or "/" in bucket or bucket.startswith("gs:"):
            raise ValueError(f"invalid input_bucket: {self.input_bucket!r}")
        prefix = self.input_prefix
        if not prefix or prefix.startswith("/"):
            raise ValueError(f"invalid input_prefix: {self.input_prefix!r}")
        parts = [p for p in prefix.split("/") if p != ""]
        if any(part in {".", ".."} for part in parts):
            raise ValueError(f"invalid input_prefix: {self.input_prefix!r}")
        if not prefix.endswith("/"):
            object.__setattr__(self, "input_prefix", prefix + "/")
        if int(self.max_pcap_bytes) <= 0:
            raise ValueError(f"max_pcap_bytes must be > 0, got {self.max_pcap_bytes}")

        mode = (self.pcap_fetcher or DEFAULT_PCAP_FETCHER).strip().lower()
        if mode not in {"gcs", "local"}:
            raise ValueError(f"invalid pcap_fetcher: {self.pcap_fetcher!r}")
        object.__setattr__(self, "pcap_fetcher", mode)
        if mode == "local":
            root = (self.local_pcap_root or "").strip()
            if not root:
                raise ValueError(
                    "local_pcap_root / IOMT_LOCAL_PCAP_ROOT is required when "
                    "pcap_fetcher=local"
                )
            object.__setattr__(self, "local_pcap_root", root)

    @classmethod
    def from_env(cls) -> ServingSettings:
        """Load from IOMT_* environment variables (with provisional defaults)."""
        return cls(
            input_bucket=os.environ.get("IOMT_INPUT_BUCKET", DEFAULT_INPUT_BUCKET),
            input_prefix=os.environ.get("IOMT_INPUT_PREFIX", DEFAULT_INPUT_PREFIX),
            max_pcap_bytes=int(
                os.environ.get("IOMT_MAX_PCAP_BYTES", str(DEFAULT_MAX_PCAP_BYTES))
            ),
            pcap_fetcher=os.environ.get("IOMT_PCAP_FETCHER", DEFAULT_PCAP_FETCHER),
            local_pcap_root=os.environ.get("IOMT_LOCAL_PCAP_ROOT") or None,
        )
