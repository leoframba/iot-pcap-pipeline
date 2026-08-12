"""Frozen HGB loader / scorer for V1 serving (process-lifetime engine)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

from iot_pcap_pipeline.paths import PROJECT_ROOT
from iot_pcap_pipeline.serving.contract import (
    EXPECTED_MODEL_SHA256,
    load_model_input_feature_names,
    load_serving_contract,
    sha256_file,
    verify_serving_contract,
)
from iot_pcap_pipeline.serving.errors import ServingError
from iot_pcap_pipeline.serving.labels import ATTACK_CLASS, BENIGN_CLASS


@dataclass(frozen=True)
class V1InferenceEngine:
    """Loaded once per process: contract + HGB + ordered 22-feature names."""

    contract: dict[str, Any]
    estimator: Any
    feature_names: tuple[str, ...]
    model_path: Path
    model_sha256: str
    attack_class_index: int

    @classmethod
    def load_default(
        cls,
        *,
        project_root: Path | None = None,
        contract_path: Path | str | None = None,
        model_path: Path | str | None = None,
    ) -> V1InferenceEngine:
        root = (project_root or PROJECT_ROOT).resolve()
        contract = verify_serving_contract(
            load_serving_contract(contract_path, project_root=root),
            project_root=root,
        )
        feature_names = tuple(load_model_input_feature_names(project_root=root))
        if list(feature_names) != list((contract.get("model") or {}).get("feature_names") or []):
            raise ServingError("engine feature_names drift vs serving contract")

        rel = (contract.get("model") or {}).get("model_artifact")
        if model_path is not None:
            path = Path(model_path)
            if not path.is_absolute():
                path = root / path
        else:
            path = root / Path(rel or "artifacts/v1/H0_full_fit.joblib")
        if not path.is_file():
            raise ServingError(f"model artifact missing: {path}")

        digest = sha256_file(path)
        expected = str((contract.get("model") or {}).get("model_artifact_sha256") or "")
        if digest != expected or digest != EXPECTED_MODEL_SHA256:
            raise ServingError(
                f"model SHA mismatch before load: actual={digest} "
                f"pinned={expected} expected={EXPECTED_MODEL_SHA256}"
            )

        estimator = joblib.load(path)
        attack_idx = _verify_estimator(estimator, n_features=len(feature_names))
        return cls(
            contract=contract,
            estimator=estimator,
            feature_names=feature_names,
            model_path=path,
            model_sha256=digest,
            attack_class_index=attack_idx,
        )

    def score_matrix(self, X: np.ndarray) -> np.ndarray:
        """Return uncalibrated window_attack_score for each row."""
        if X.ndim != 2:
            raise ServingError(f"expected 2D feature matrix, got ndim={X.ndim}")
        if X.shape[1] != len(self.feature_names):
            raise ServingError(
                f"feature width {X.shape[1]} != {len(self.feature_names)}"
            )
        if X.shape[0] == 0:
            return np.asarray([], dtype=np.float64)
        if not np.isfinite(X).all():
            raise ServingError("non-finite values in model input")
        proba = self.estimator.predict_proba(X)
        return np.asarray(proba[:, self.attack_class_index], dtype=np.float64)

    def score_rows(self, rows: Sequence[Sequence[float]]) -> np.ndarray:
        X = np.asarray(rows, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.score_matrix(X)


def _verify_estimator(estimator: Any, *, n_features: int) -> int:
    if not hasattr(estimator, "predict_proba"):
        raise ServingError(f"estimator lacks predict_proba: {type(estimator)!r}")
    n_in = getattr(estimator, "n_features_in_", None)
    if n_in is not None and int(n_in) != n_features:
        raise ServingError(f"n_features_in_={n_in} != {n_features}")

    classes = list(getattr(estimator, "classes_", []))
    if not classes and hasattr(estimator, "named_steps"):
        final = estimator.named_steps[list(estimator.named_steps)[-1]]
        classes = list(getattr(final, "classes_", []))
    if BENIGN_CLASS not in classes or ATTACK_CLASS not in classes:
        raise ServingError(
            f"estimator.classes_ missing BENIGN/ATTACK ids: {classes!r}"
        )
    return int(classes.index(ATTACK_CLASS))
