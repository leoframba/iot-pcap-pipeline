"""GCS URI parsing and PCAP fetchers (production + test doubles)."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol


class PcapFetchError(ValueError):
    """Raised when a GCS URI is rejected or a download fails policy checks."""


@dataclass(frozen=True)
class GcsObjectRef:
    bucket: str
    object_name: str

    @property
    def uri(self) -> str:
        return f"gs://{self.bucket}/{self.object_name}"


def parse_gcs_uri(gcs_uri: str) -> GcsObjectRef:
    """Parse ``gs://bucket/object`` into bucket + object name."""
    if not isinstance(gcs_uri, str) or not gcs_uri.startswith("gs://"):
        raise PcapFetchError("gcs_uri must start with gs://")
    rest = gcs_uri[len("gs://") :]
    if not rest or "/" not in rest:
        raise PcapFetchError("gcs_uri must include bucket and object name")
    bucket, object_name = rest.split("/", 1)
    if not bucket or not object_name:
        raise PcapFetchError("gcs_uri must include bucket and object name")
    if "\\" in object_name or object_name.startswith("/"):
        raise PcapFetchError(f"rejected gcs object path: {object_name!r}")
    parts = object_name.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise PcapFetchError(f"rejected gcs object path: {object_name!r}")
    return GcsObjectRef(bucket=bucket, object_name=object_name)


def ensure_uri_allowed(ref: GcsObjectRef, *, input_bucket: str, input_prefix: str) -> None:
    """Application allowlist: exact bucket + object prefix (in addition to IAM)."""
    if ref.bucket != input_bucket:
        raise PcapFetchError(
            f"gcs bucket not allowed: {ref.bucket!r} (expected {input_bucket!r})"
        )
    if not ref.object_name.startswith(input_prefix):
        raise PcapFetchError(
            f"gcs object prefix not allowed: {ref.object_name!r} "
            f"(expected prefix {input_prefix!r})"
        )


def ensure_size_allowed(size_bytes: int, *, max_pcap_bytes: int, where: str) -> None:
    if size_bytes < 0:
        raise PcapFetchError(f"invalid object size ({where}): {size_bytes}")
    if size_bytes > max_pcap_bytes:
        raise PcapFetchError(
            f"PCAP too large ({where}): {size_bytes} bytes > max {max_pcap_bytes}"
        )


class PcapFetcher(Protocol):
    """gs:// URI → temporary local PCAP path."""

    def fetch(self, gcs_uri: str, destination: Path) -> Path:
        """Download (or materialize) ``gcs_uri`` to ``destination`` and return it."""


class GcsPcapFetcher:
    """Production fetcher using google-cloud-storage + ADC."""

    def __init__(self, *, input_bucket: str, input_prefix: str, max_pcap_bytes: int, client=None):
        self.input_bucket = input_bucket
        self.input_prefix = input_prefix
        self.max_pcap_bytes = int(max_pcap_bytes)
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from google.cloud import storage  # lazy: keep import off health-only paths

            self._client = storage.Client()
        return self._client

    def fetch(self, gcs_uri: str, destination: Path) -> Path:
        ref = parse_gcs_uri(gcs_uri)
        ensure_uri_allowed(
            ref, input_bucket=self.input_bucket, input_prefix=self.input_prefix
        )
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        blob = self.client.bucket(ref.bucket).blob(ref.object_name)
        blob.reload()
        size = int(blob.size) if blob.size is not None else -1
        ensure_size_allowed(size, max_pcap_bytes=self.max_pcap_bytes, where="metadata")

        blob.download_to_filename(str(destination))
        actual = destination.stat().st_size
        try:
            ensure_size_allowed(
                actual, max_pcap_bytes=self.max_pcap_bytes, where="downloaded"
            )
        except PcapFetchError:
            destination.unlink(missing_ok=True)
            raise
        return destination


class FakePcapFetcher:
    """Test double: map fake ``gs://`` URIs to local PCAP paths (no GCP credentials)."""

    def __init__(
        self,
        mapping: Mapping[str, Path | str],
        *,
        input_bucket: str,
        input_prefix: str,
        max_pcap_bytes: int,
    ):
        self.mapping = {str(k): Path(v) for k, v in mapping.items()}
        self.input_bucket = input_bucket
        self.input_prefix = input_prefix
        self.max_pcap_bytes = int(max_pcap_bytes)

    def fetch(self, gcs_uri: str, destination: Path) -> Path:
        ref = parse_gcs_uri(gcs_uri)
        ensure_uri_allowed(
            ref, input_bucket=self.input_bucket, input_prefix=self.input_prefix
        )
        if gcs_uri not in self.mapping:
            raise PcapFetchError(f"fake GCS object not found: {gcs_uri}")
        src = self.mapping[gcs_uri]
        if not src.is_file():
            raise PcapFetchError(f"fake GCS source missing on disk: {src}")

        size = src.stat().st_size
        ensure_size_allowed(size, max_pcap_bytes=self.max_pcap_bytes, where="metadata")

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, destination)
        actual = destination.stat().st_size
        try:
            ensure_size_allowed(
                actual, max_pcap_bytes=self.max_pcap_bytes, where="downloaded"
            )
        except PcapFetchError:
            destination.unlink(missing_ok=True)
            raise
        return destination
