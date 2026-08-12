"""D1 local PCAP inference core: decode → window → extract → score → aggregate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import dpkt
import numpy as np

from iot_pcap_pipeline.features.extractor import FeatureVector, extract_features
from iot_pcap_pipeline.features.schema import V1_FEATURE_NAMES
from iot_pcap_pipeline.pcap.decode import DLT_EN10MB
from iot_pcap_pipeline.pcap.reader import iter_packets
from iot_pcap_pipeline.serving.aggregate import (
    AggregationResult,
    StreamingWindowAggregator,
)
from iot_pcap_pipeline.serving.contract import (
    ACCEPTED_LINKTYPE,
    WINDOW_ATTACK_THRESHOLD,
)
from iot_pcap_pipeline.serving.errors import (
    STATUS_INVALID_INPUT,
    STATUS_UNSUPPORTED_INPUT,
    ServingError,
)
from iot_pcap_pipeline.serving.model import V1InferenceEngine
from iot_pcap_pipeline.windowing.stream import FeatureExtractionError, iter_windows

# Implementation parameter only (not a serving semantic).
DEFAULT_SCORE_BATCH_SIZE = 1024


@dataclass(frozen=True)
class ClassifyResult:
    status: str
    prediction: str | None
    pcap_attack_score: float | None
    window_summary: dict[str, Any]
    decision: dict[str, Any]
    model: dict[str, Any]
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {
            "status": self.status,
            "prediction": self.prediction,
            "pcap_attack_score": self.pcap_attack_score,
            "window_summary": dict(self.window_summary),
            "decision": dict(self.decision),
            "model": dict(self.model),
        }
        if self.detail is not None:
            out["detail"] = self.detail
        return out


def peek_pcap_linktype(pcap_path: Path | str) -> int:
    """Read libpcap datalink type without scoring packets."""
    path = Path(pcap_path)
    if not path.is_file():
        raise FileNotFoundError(f"PCAP not found: {path}")
    with path.open("rb") as handle:
        try:
            reader = dpkt.pcap.Reader(handle)
        except (ValueError, OSError, dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError) as exc:
            raise ValueError(f"failed to open PCAP {path}: {exc}") from exc
        return int(reader.datalink())


def select_model_features_22(
    features27: FeatureVector,
    feature_names_22: list[str] | tuple[str, ...],
) -> list[float]:
    """Select exact frozen 22 model features from a 27-feature vector."""
    data = features27.to_feature_dict()
    if len(V1_FEATURE_NAMES) != 27:
        raise ServingError("V1_FEATURE_NAMES length drift")
    try:
        return [float(data[name]) for name in feature_names_22]
    except KeyError as exc:
        raise ServingError(f"missing model feature in extractor output: {exc}") from exc


def _empty_window_summary() -> dict[str, Any]:
    return {
        "total_windows": 0,
        "attack_windows": 0,
        "benign_windows": 0,
        "max_window_attack_score": None,
        "mean_window_attack_score": None,
    }


def _model_block(engine: V1InferenceEngine) -> dict[str, Any]:
    return {
        "model_version": (engine.contract.get("model") or {}).get("model_version"),
        "serving_contract_version": engine.contract.get("serving_contract_version"),
        "score_semantics": "uncalibrated_model_score",
        "model_artifact_sha256": engine.model_sha256,
    }


def _decision_block(engine: V1InferenceEngine) -> dict[str, Any]:
    pcap = engine.contract.get("pcap_decision") or {}
    window = engine.contract.get("window_decision") or {}
    return {
        "window_attack_threshold": float(
            window.get("window_attack_threshold", WINDOW_ATTACK_THRESHOLD)
        ),
        "minimum_complete_windows": int(pcap.get("minimum_complete_windows")),
        "pcap_min_attack_windows": int(pcap.get("pcap_min_attack_windows")),
        "pcap_attack_rate_threshold": float(pcap.get("pcap_attack_rate_threshold")),
    }


def _non_ok(
    *,
    status: str,
    engine: V1InferenceEngine,
    detail: str,
) -> ClassifyResult:
    return ClassifyResult(
        status=status,
        prediction=None,
        pcap_attack_score=None,
        window_summary=_empty_window_summary(),
        decision=_decision_block(engine),
        model=_model_block(engine),
        detail=detail,
    )


def _from_aggregation(
    agg: AggregationResult,
    *,
    engine: V1InferenceEngine,
) -> ClassifyResult:
    return ClassifyResult(
        status=agg.status,
        prediction=agg.prediction,
        pcap_attack_score=agg.pcap_attack_score,
        window_summary=asdict(agg.window_summary),
        decision=asdict(agg.decision),
        model=_model_block(engine),
    )


def classify_pcap(
    pcap_path: Path | str,
    *,
    engine: V1InferenceEngine | None = None,
    project_root: Path | None = None,
    batch_size: int = DEFAULT_SCORE_BATCH_SIZE,
) -> ClassifyResult:
    """Run frozen V1 PCAP inference (no HTTP). Reuse ``engine`` across calls."""
    if batch_size < 1:
        raise ServingError(f"batch_size must be >= 1, got {batch_size}")

    eng = engine or V1InferenceEngine.load_default(project_root=project_root)
    path = Path(pcap_path)

    try:
        linktype = peek_pcap_linktype(path)
    except FileNotFoundError as exc:
        return _non_ok(status=STATUS_INVALID_INPUT, engine=eng, detail=str(exc))
    except ValueError as exc:
        return _non_ok(status=STATUS_INVALID_INPUT, engine=eng, detail=str(exc))

    if int(linktype) != int(ACCEPTED_LINKTYPE) or int(linktype) != int(DLT_EN10MB):
        return _non_ok(
            status=STATUS_UNSUPPORTED_INPUT,
            engine=eng,
            detail=f"unsupported linktype={linktype}; V1 accepts DLT_EN10MB=1 only",
        )

    feature_names_22 = list(eng.feature_names)
    pcap_dec = eng.contract.get("pcap_decision") or {}
    aggregator = StreamingWindowAggregator(
        window_threshold=float(
            (eng.contract.get("window_decision") or {}).get(
                "window_attack_threshold", WINDOW_ATTACK_THRESHOLD
            )
        ),
        minimum_complete_windows=int(pcap_dec.get("minimum_complete_windows")),
        min_attack_windows=int(pcap_dec.get("pcap_min_attack_windows")),
        attack_rate_threshold=float(pcap_dec.get("pcap_attack_rate_threshold")),
    )

    batch: list[list[float]] = []

    def flush() -> None:
        nonlocal batch
        if not batch:
            return
        X = np.asarray(batch, dtype=np.float32)
        scores = eng.score_matrix(X)
        aggregator.observe_many([float(s) for s in scores.tolist()])
        batch = []

    try:
        for window in iter_windows(iter_packets(path)):
            fv = extract_features(window)
            batch.append(select_model_features_22(fv, feature_names_22))
            if len(batch) >= batch_size:
                flush()
        flush()
    except FeatureExtractionError as exc:
        return _non_ok(status=STATUS_INVALID_INPUT, engine=eng, detail=str(exc))
    except (ValueError, OSError) as exc:
        return _non_ok(status=STATUS_INVALID_INPUT, engine=eng, detail=str(exc))

    return _from_aggregation(aggregator.finalize(), engine=eng)
