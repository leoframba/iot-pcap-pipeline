"""Predeclared D0 PCAP aggregation candidate policies (TRAIN-validation only)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iot_pcap_pipeline.modeling.baselines.phase2c_freeze import FROZEN_V1_THRESHOLD
from iot_pcap_pipeline.paths import PROJECT_ROOT

DEFAULT_CANDIDATES_PATH = (
    PROJECT_ROOT / "data" / "serving" / "v1" / "pcap_aggregation_candidates.json"
)
DEFAULT_DRAFT_CONTRACT_PATH = (
    PROJECT_ROOT / "data" / "serving" / "v1" / "serving_contract_draft.json"
)

# Locked window operating point (must match v1_model_package.json).
WINDOW_ATTACK_THRESHOLD = FROZEN_V1_THRESHOLD

K_CANDIDATES: tuple[int, ...] = (1, 3, 5)
R_CANDIDATES: tuple[float, ...] = (0.005, 0.01, 0.025, 0.05)


@dataclass(frozen=True)
class AggregationPolicy:
    """One predeclared (K, R) PCAP decision policy."""

    policy_id: str
    min_attack_windows: int  # K
    attack_rate_threshold: float  # R

    @property
    def K(self) -> int:
        return self.min_attack_windows

    @property
    def R(self) -> float:
        return self.attack_rate_threshold


def policy_id_for(K: int, R: float) -> str:
    """Stable id matching pcap_aggregation_candidates.json."""
    r_txt = f"{R:g}"
    return f"K{K}_R{r_txt}"


def iter_candidate_policies() -> tuple[AggregationPolicy, ...]:
    """Return the frozen 12-policy grid in declaration order."""
    policies: list[AggregationPolicy] = []
    for K in K_CANDIDATES:
        for R in R_CANDIDATES:
            policies.append(
                AggregationPolicy(
                    policy_id=policy_id_for(K, R),
                    min_attack_windows=K,
                    attack_rate_threshold=R,
                )
            )
    return tuple(policies)


def load_candidates_document(
    path: Path | str | None = None,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = (project_root or PROJECT_ROOT).resolve()
    p = Path(path or DEFAULT_CANDIDATES_PATH)
    if not p.is_absolute():
        p = root / p
    return json.loads(p.read_text(encoding="utf-8"))


def verify_candidates_document(doc: dict[str, Any]) -> None:
    """Refuse drift between JSON candidates and in-code grid."""
    policies = doc.get("policies") or []
    expected = iter_candidate_policies()
    if len(policies) != len(expected):
        raise ValueError(
            f"candidate policy count {len(policies)} != {len(expected)}"
        )
    for raw, exp in zip(policies, expected, strict=True):
        if raw.get("policy_id") != exp.policy_id:
            raise ValueError(
                f"policy_id mismatch: {raw.get('policy_id')!r} != {exp.policy_id!r}"
            )
        if int(raw["K"]) != exp.K or float(raw["R"]) != exp.R:
            raise ValueError(f"K/R mismatch for {exp.policy_id}")
    thr = float((doc.get("window_decision") or {}).get("window_attack_threshold"))
    if thr != WINDOW_ATTACK_THRESHOLD:
        raise ValueError(
            f"window threshold {thr!r} != frozen {WINDOW_ATTACK_THRESHOLD!r}"
        )


SELECTION_PRIORITY: tuple[str, ...] = (
    "Minimize benign-PCAP false positives.",
    "Among remaining policies, maximize attack-PCAP recall.",
    "Break ties using macro family recall.",
    "If still tied, prefer lower K then lower R.",
)
