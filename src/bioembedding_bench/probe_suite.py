from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import classification_metrics
from .splits import grouped_splits


def _probe(kind: str, seed: int):
    if kind == "linear":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )
    elif kind == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-3,
            early_stopping=True,
            validation_fraction=0.15,
            max_iter=600,
            random_state=seed,
        )
    else:
        raise ValueError("kind must be 'linear' or 'mlp'")
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def compare_probes(
    X,
    y,
    groups,
    n_splits: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Compare linear and nonlinear probes using identical group-held-out folds."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)

    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if not (len(X) == len(y) == len(groups)):
        raise ValueError("X, y, and groups must have matching lengths")

    folds = list(grouped_splits(y, groups, n_splits=n_splits, seed=seed))
    rows = []

    for kind in ("linear", "mlp"):
        for fold, (train_idx, test_idx) in enumerate(folds, start=1):
            if set(groups[train_idx]) & set(groups[test_idx]):
                raise AssertionError("group leakage detected")

            model = _probe(kind, seed + fold)
            model.fit(X[train_idx], y[train_idx])
            probability = model.predict_proba(X[test_idx])[:, 1]

            rows.append(
                {
                    "probe": kind,
                    "fold": fold,
                    "train_groups": int(len(np.unique(groups[train_idx]))),
                    "test_groups": int(len(np.unique(groups[test_idx]))),
                    **classification_metrics(y[test_idx], probability),
                }
            )

    return pd.DataFrame(rows)
