"""Unit tests for GCS URI parsing, allowlist, and size guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from iot_pcap_pipeline.api.settings import ServingSettings
from iot_pcap_pipeline.api.storage import (
    FakePcapFetcher,
    GcsNotAllowedError,
    GcsNotFoundError,
    GcsPermissionDeniedError,
    GcsUriInvalidError,
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
