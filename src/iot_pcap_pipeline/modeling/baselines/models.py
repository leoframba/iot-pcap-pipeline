"""Fixed unweighted baseline model definitions for Phase 2B.2."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from iot_pcap_pipeline.modeling.baselines.constants import LABEL_MAPPING

RANDOM_SEED = 42

LOGISTIC_PARAMS: dict[str, Any] = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "class_weight": None,
}

HGB_PARAMS: dict[str, Any] = {
    "learning_rate": 0.1,
    "max_iter": 200,
    "max_leaf_nodes": 31,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": RANDOM_SEED,
}


def build_logistic_regression() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(**LOGISTIC_PARAMS)),
        ]
    )


def build_hist_gradient_boosting() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**HGB_PARAMS)


def attack_score_from_estimator(estimator: Any, X) -> Any:
    """Return P(ATTACK) using frozen label mapping; refuse ambiguous class order."""
    import numpy as np

    if not hasattr(estimator, "predict_proba"):
        raise TypeError(f"estimator lacks predict_proba: {type(estimator)!r}")
    proba = estimator.predict_proba(X)
    classes = list(getattr(estimator, "classes_", []))
    # Pipelines expose classes_ on the final step after fit.
    if not classes and hasattr(estimator, "named_steps"):
        final = estimator.named_steps[list(estimator.named_steps)[-1]]
        classes = list(getattr(final, "classes_", []))
    attack = LABEL_MAPPING["ATTACK"]
    benign = LABEL_MAPPING["BENIGN"]
    if classes == [benign, attack]:
        return np.asarray(proba[:, 1], dtype=np.float32)
    if attack not in classes:
        raise RuntimeError(
            f"ATTACK class {attack} missing from estimator.classes_={classes!r}"
        )
    idx = classes.index(attack)
    return np.asarray(proba[:, idx], dtype=np.float32)


MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "model_id": "logistic_regression",
        "display_name": "StandardScaler + LogisticRegression",
        "builder": build_logistic_regression,
        "hyperparameters": {
            "pipeline": ["StandardScaler", "LogisticRegression"],
            "logistic_regression": dict(LOGISTIC_PARAMS),
        },
    },
    {
        "model_id": "hist_gradient_boosting",
        "display_name": "HistGradientBoostingClassifier",
        "builder": build_hist_gradient_boosting,
        "hyperparameters": dict(HGB_PARAMS),
    },
)
