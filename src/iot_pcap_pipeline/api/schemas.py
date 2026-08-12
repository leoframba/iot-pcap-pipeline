"""Pydantic models for the V1 Vertex-style predict HTTP schema."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class PredictInstance(BaseModel):
    """Single prediction instance: GCS object reference only."""

    gcs_uri: str = Field(..., min_length=1)

    @field_validator("gcs_uri")
    @classmethod
    def gcs_uri_must_be_gs(cls, value: str) -> str:
        if not value.startswith("gs://"):
            raise ValueError("gcs_uri must start with gs://")
        rest = value[len("gs://") :]
        if "/" not in rest or not rest.split("/", 1)[0] or not rest.split("/", 1)[1]:
            raise ValueError("gcs_uri must include bucket and object name")
        return value


class PredictRequest(BaseModel):
    """Vertex-compatible predict body: exactly one instance."""

    instances: list[PredictInstance]

    @model_validator(mode="after")
    def exactly_one_instance(self) -> PredictRequest:
        if len(self.instances) != 1:
            raise ValueError("instances must contain exactly one element")
        return self


class WindowSummary(BaseModel):
    total_windows: int
    attack_windows: int
    benign_windows: int
    max_window_attack_score: float | None
    mean_window_attack_score: float | None


class DecisionBlock(BaseModel):
    window_attack_threshold: float
    minimum_complete_windows: int
    pcap_min_attack_windows: int
    pcap_attack_rate_threshold: float


class ModelBlock(BaseModel):
    model_version: str
    serving_contract_version: str
    score_semantics: Literal["uncalibrated_model_score"]


class PredictionItem(BaseModel):
    """Contract-shaped classify result (plus optional detail)."""

    status: str
    prediction: str | None
    pcap_attack_score: float | None
    window_summary: WindowSummary
    decision: DecisionBlock
    model: ModelBlock
    detail: str | None = None


class PredictResponse(BaseModel):
    predictions: list[PredictionItem]

    @classmethod
    def from_classify_dict(cls, payload: dict[str, Any]) -> PredictResponse:
        """Wrap ``ClassifyResult.to_dict()`` with no score transformation."""
        return cls(predictions=[PredictionItem.model_validate(payload)])
