from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_linear_probe(seed: int = 0, c: float = 1.0) -> Pipeline:
    """A deliberately simple probe for measuring linear accessibility."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=c,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )


def fit_probability(X_train, y_train, X_test, seed: int = 0):
    model = make_linear_probe(seed=seed)
    model.fit(np.asarray(X_train, dtype=float), np.asarray(y_train, dtype=int))
    return model.predict_proba(np.asarray(X_test, dtype=float))[:, 1]
