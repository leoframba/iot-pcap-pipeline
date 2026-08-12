"""Frozen serving label / class-id pins (deployment runtime)."""

from __future__ import annotations

# Explicit training label mapping used by the frozen HGB estimator.
BENIGN_CLASS = 0
ATTACK_CLASS = 1

LABEL_NAMES: dict[int, str] = {
    BENIGN_CLASS: "BENIGN",
    ATTACK_CLASS: "ATTACK",
}
