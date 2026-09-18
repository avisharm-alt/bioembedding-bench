from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score


def classification_metrics(y_true, probability) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= 0.5).astype(int)

    auc = (
        float(roc_auc_score(y_true, probability))
        if len(np.unique(y_true)) == 2
        else float("nan")
    )
    return {
        "roc_auc": auc,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "brier": float(brier_score_loss(y_true, probability)),
    }
