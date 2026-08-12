"""Unit tests for serving estimator class-order verification."""

from __future__ import annotations

import numpy as np
import pytest

from iot_pcap_pipeline.serving.errors import ServingError
from iot_pcap_pipeline.serving.labels import ATTACK_CLASS, BENIGN_CLASS
from iot_pcap_pipeline.serving.model import _verify_estimator


class _FakeEstimator:
    def __init__(
        self,
        *,
        classes: list[int] | None,
        n_features_in: int | None = 22,
        has_predict_proba: bool = True,
    ) -> None:
        if classes is not None:
            self.classes_ = np.asarray(classes)
        if n_features_in is not None:
            self.n_features_in_ = n_features_in
        if has_predict_proba:
            self.predict_proba = lambda X: np.zeros((len(X), 2), dtype=np.float64)


def test_verify_estimator_classes_0_1_attack_index_1() -> None:
    assert _verify_estimator(_FakeEstimator(classes=[0, 1]), n_features=22) == 1


def test_verify_estimator_classes_1_0_attack_index_0() -> None:
    assert _verify_estimator(_FakeEstimator(classes=[1, 0]), n_features=22) == 0


def test_verify_estimator_missing_attack_class_rejects() -> None:
    with pytest.raises(ServingError, match="classes_"):
        _verify_estimator(_FakeEstimator(classes=[0, 2]), n_features=22)


def test_verify_estimator_missing_predict_proba_rejects() -> None:
    with pytest.raises(ServingError, match="predict_proba"):
        _verify_estimator(
            _FakeEstimator(classes=[0, 1], has_predict_proba=False),
            n_features=22,
        )


def test_verify_estimator_wrong_n_features_rejects() -> None:
    with pytest.raises(ServingError, match="n_features_in_"):
        _verify_estimator(_FakeEstimator(classes=[0, 1], n_features_in=27), n_features=22)


def test_verify_estimator_requires_both_class_ids() -> None:
    with pytest.raises(ServingError, match="classes_"):
        _verify_estimator(_FakeEstimator(classes=[ATTACK_CLASS]), n_features=22)
    with pytest.raises(ServingError, match="classes_"):
        _verify_estimator(_FakeEstimator(classes=[BENIGN_CLASS]), n_features=22)
