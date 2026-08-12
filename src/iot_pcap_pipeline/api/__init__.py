"""HTTP serving surface (FastAPI). Requires the ``serving`` optional extra."""

from __future__ import annotations

from iot_pcap_pipeline.api.app import MAX_CONCURRENT_PREDICTIONS, create_app
from iot_pcap_pipeline.api.settings import ServingSettings
from iot_pcap_pipeline.api.storage import (
    FakePcapFetcher,
    GcsPcapFetcher,
    PcapFetchError,
)

__all__ = [
    "FakePcapFetcher",
    "GcsPcapFetcher",
    "MAX_CONCURRENT_PREDICTIONS",
    "PcapFetchError",
    "ServingSettings",
    "create_app",
]
