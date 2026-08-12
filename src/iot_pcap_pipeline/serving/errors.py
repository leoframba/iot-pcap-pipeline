"""Serving-side errors and result statuses (no research imports)."""

from __future__ import annotations


class ServingError(RuntimeError):
    """Raised when a serving contract / artifact check fails."""


STATUS_OK = "OK"
STATUS_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
STATUS_INVALID_INPUT = "INVALID_INPUT"
STATUS_UNSUPPORTED_INPUT = "UNSUPPORTED_INPUT"
