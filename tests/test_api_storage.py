"""Unit tests for GCS URI parsing, allowlist, and size guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from iot_pcap_pipeline.api.settings import ServingSettings
from iot_pcap_pipeline.api.storage import (
    FakePcapFetcher,
    GcsNotAllowedError,
    GcsNotFoundError,
    GcsPcapFetcher,
    GcsPermissionDeniedError,
    GcsUriInvalidError,
    PcapFetchError,
    PcapTooLargeError,
    ensure_uri_allowed,
    parse_gcs_uri,
)


def test_parse_gcs_uri_ok() -> None:
    ref = parse_gcs_uri("gs://iomt-input/pcaps/example.pcap")
    assert ref.bucket == "iomt-input"
    assert ref.object_name == "pcaps/example.pcap"


@pytest.mark.parametrize(
    "uri",
    [
        "http://iomt-input/pcaps/x.pcap",
        "gs://iomt-input",
        "gs://iomt-input/",
        "gs:///pcaps/x.pcap",
        "not-a-uri",
        "",
    ],
)
def test_parse_gcs_uri_rejects_malformed(uri: str) -> None:
    with pytest.raises(GcsUriInvalidError) as excinfo:
        parse_gcs_uri(uri)
    assert excinfo.value.status_code == 422


def test_ensure_uri_allowed_bucket_and_prefix() -> None:
    ref = parse_gcs_uri("gs://iomt-input/pcaps/foo.pcap")
    ensure_uri_allowed(ref, input_bucket="iomt-input", input_prefix="pcaps/")

    with pytest.raises(GcsNotAllowedError) as bucket_exc:
        ensure_uri_allowed(ref, input_bucket="other", input_prefix="pcaps/")
    assert bucket_exc.value.status_code == 403

    with pytest.raises(GcsNotAllowedError) as prefix_exc:
        ensure_uri_allowed(
            parse_gcs_uri("gs://iomt-input/private/foo"),
            input_bucket="iomt-input",
            input_prefix="pcaps/",
        )
    assert prefix_exc.value.status_code == 403


def test_fake_fetcher_enforces_size(tmp_path: Path) -> None:
    src = tmp_path / "big.pcap"
    src.write_bytes(b"x" * 100)
    settings = ServingSettings(
        input_bucket="iomt-input", input_prefix="pcaps/", max_pcap_bytes=50
    )
    fetcher = FakePcapFetcher(
        {"gs://iomt-input/pcaps/big.pcap": src},
        input_bucket=settings.input_bucket,
        input_prefix=settings.input_prefix,
        max_pcap_bytes=settings.max_pcap_bytes,
    )
    with pytest.raises(PcapTooLargeError) as excinfo:
        fetcher.fetch("gs://iomt-input/pcaps/big.pcap", tmp_path / "out.pcap")
    assert excinfo.value.status_code == 413


def test_fake_fetcher_not_found_and_denied(tmp_path: Path) -> None:
    src = tmp_path / "ok.pcap"
    src.write_bytes(b"pcap-bytes")
    uri = "gs://iomt-input/pcaps/ok.pcap"
    missing = "gs://iomt-input/pcaps/missing.pcap"
    fetcher = FakePcapFetcher(
        {uri: src},
        input_bucket="iomt-input",
        input_prefix="pcaps/",
        max_pcap_bytes=1024,
        denied_uris=[uri],
    )
    with pytest.raises(GcsPermissionDeniedError):
        fetcher.fetch(uri, tmp_path / "denied.pcap")
    with pytest.raises(GcsNotFoundError):
        fetcher.fetch(missing, tmp_path / "missing.pcap")


def test_fake_fetcher_copies_allowed_object(tmp_path: Path) -> None:
    src = tmp_path / "ok.pcap"
    src.write_bytes(b"pcap-bytes")
    uri = "gs://iomt-input/pcaps/ok.pcap"
    fetcher = FakePcapFetcher(
        {uri: src},
        input_bucket="iomt-input",
        input_prefix="pcaps/",
        max_pcap_bytes=1024,
    )
    dest = tmp_path / "downloaded.pcap"
    out = fetcher.fetch(uri, dest)
    assert out == dest
    assert dest.read_bytes() == b"pcap-bytes"
    assert fetcher.last_destination == dest


def test_settings_normalizes_prefix_slash() -> None:
    settings = ServingSettings(
        input_bucket="iomt-input", input_prefix="pcaps", max_pcap_bytes=10
    )
    assert settings.input_prefix == "pcaps/"


# --- GcsPcapFetcher with injected fake GCS client (no credentials) ---


# Exception class *names* must match google.api_core.exceptions for translation.
NotFound = type("NotFound", (Exception,), {})
Forbidden = type("Forbidden", (Exception,), {})


class _FakeBlob:
    def __init__(
        self,
        *,
        size: int | None = 0,
        payload: bytes = b"",
        reload_error: BaseException | None = None,
        download_error: BaseException | None = None,
        download_payload: bytes | None = None,
    ) -> None:
        self.size = size
        self.payload = payload
        self.reload_error = reload_error
        self.download_error = download_error
        self.download_payload = download_payload
        self.reload_calls = 0
        self.download_calls = 0

    def reload(self) -> None:
        self.reload_calls += 1
        if self.reload_error is not None:
            raise self.reload_error

    def download_to_filename(self, filename: str) -> None:
        self.download_calls += 1
        path = Path(filename)
        if self.download_error is not None:
            # Simulate a partial write before failure.
            path.write_bytes(b"partial")
            raise self.download_error
        data = self.payload if self.download_payload is None else self.download_payload
        path.write_bytes(data)


class _FakeBucket:
    def __init__(self, blob: _FakeBlob) -> None:
        self._blob = blob
        self.blob_names: list[str] = []

    def blob(self, name: str) -> _FakeBlob:
        self.blob_names.append(name)
        return self._blob


class _FakeClient:
    def __init__(self, blob: _FakeBlob) -> None:
        self._bucket = _FakeBucket(blob)
        self.bucket_names: list[str] = []

    def bucket(self, name: str) -> _FakeBucket:
        self.bucket_names.append(name)
        return self._bucket


def _gcs_fetcher(client: _FakeClient, *, max_pcap_bytes: int = 1024) -> GcsPcapFetcher:
    return GcsPcapFetcher(
        input_bucket="iomt-input",
        input_prefix="pcaps/",
        max_pcap_bytes=max_pcap_bytes,
        client=client,
    )


def test_gcs_fetcher_downloads_when_metadata_size_ok(tmp_path: Path) -> None:
    blob = _FakeBlob(size=11, payload=b"hello-pcap!")
    client = _FakeClient(blob)
    fetcher = _gcs_fetcher(client)
    dest = tmp_path / "out.pcap"
    out = fetcher.fetch("gs://iomt-input/pcaps/ok.pcap", dest)
    assert out == dest
    assert dest.read_bytes() == b"hello-pcap!"
    assert blob.reload_calls == 1
    assert blob.download_calls == 1
    assert client.bucket_names == ["iomt-input"]
    assert client._bucket.blob_names == ["pcaps/ok.pcap"]


def test_gcs_fetcher_metadata_too_large_skips_download(tmp_path: Path) -> None:
    blob = _FakeBlob(size=10_000, payload=b"x" * 10_000)
    client = _FakeClient(blob)
    fetcher = _gcs_fetcher(client, max_pcap_bytes=100)
    with pytest.raises(PcapTooLargeError) as excinfo:
        fetcher.fetch("gs://iomt-input/pcaps/big.pcap", tmp_path / "out.pcap")
    assert excinfo.value.status_code == 413
    assert blob.reload_calls == 1
    assert blob.download_calls == 0
    assert not (tmp_path / "out.pcap").exists()


def test_gcs_fetcher_reload_not_found_maps_404(tmp_path: Path) -> None:
    blob = _FakeBlob(reload_error=NotFound("404 missing object"))
    client = _FakeClient(blob)
    fetcher = _gcs_fetcher(client)
    with pytest.raises(GcsNotFoundError) as excinfo:
        fetcher.fetch("gs://iomt-input/pcaps/missing.pcap", tmp_path / "out.pcap")
    assert excinfo.value.status_code == 404
    assert blob.download_calls == 0


def test_gcs_fetcher_reload_forbidden_maps_403(tmp_path: Path) -> None:
    blob = _FakeBlob(reload_error=Forbidden("403 denied"))
    client = _FakeClient(blob)
    fetcher = _gcs_fetcher(client)
    with pytest.raises(GcsPermissionDeniedError) as excinfo:
        fetcher.fetch("gs://iomt-input/pcaps/denied.pcap", tmp_path / "out.pcap")
    assert excinfo.value.status_code == 403
    assert blob.download_calls == 0


def test_gcs_fetcher_download_failure_removes_partial_file(tmp_path: Path) -> None:
    blob = _FakeBlob(size=10, download_error=RuntimeError("boom mid-download"))
    client = _FakeClient(blob)
    fetcher = _gcs_fetcher(client)
    dest = tmp_path / "out.pcap"
    with pytest.raises(PcapFetchError) as excinfo:
        fetcher.fetch("gs://iomt-input/pcaps/fail.pcap", dest)
    assert excinfo.value.status_code == 500
    assert "boom mid-download" in str(excinfo.value)
    assert blob.download_calls == 1
    assert not dest.exists()


def test_gcs_fetcher_post_download_oversize_removes_file(tmp_path: Path) -> None:
    # Metadata claims small; download writes larger than max.
    blob = _FakeBlob(size=10, download_payload=b"x" * 200)
    client = _FakeClient(blob)
    fetcher = _gcs_fetcher(client, max_pcap_bytes=50)
    dest = tmp_path / "out.pcap"
    with pytest.raises(PcapTooLargeError) as excinfo:
        fetcher.fetch("gs://iomt-input/pcaps/lie.pcap", dest)
    assert excinfo.value.status_code == 413
    assert "downloaded" in str(excinfo.value)
    assert blob.download_calls == 1
    assert not dest.exists()
