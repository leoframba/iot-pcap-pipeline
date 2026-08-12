"""HTTP serving surface (FastAPI). Requires the ``serving`` optional extra."""

from __future__ import annotations

from iot_pcap_pipeline.api.app import create_app

__all__ = ["create_app"]
