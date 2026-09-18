from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import classification_metrics
from .probes import fit_probability
from .splits import grouped_splits, leave_one_domain_out, naive_split


@dataclass
class BenchmarkReport:
    grouped_folds: pd.DataFrame
    domain_shift: pd.DataFrame
    summary: dict[str, float]


def _assert_disjoint(train_groups, test_groups):
    overlap = set(train_groups) & set(test_groups)
    if overlap:
        raise AssertionError(f"group leakage detected: {list(overlap)[:5]}")


def benchmark_embeddings(
    X,
    y,
    groups,
    domains=None,
    n_splits: int = 5,
    seed: int = 0,
) -> BenchmarkReport:
    """Benchmark a frozen representation under group and domain shifts."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if not (len(X) == len(y) == len(groups)):
        raise ValueError("X, y, and groups must have matching lengths")
    if len(np.unique(y)) != 2:
        raise ValueError("binary labels are required")

    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(
        grouped_splits(y, groups, n_splits=n_splits, seed=seed), start=1
    ):
        _assert_disjoint(groups[train_idx], groups[test_idx])
        p = fit_probability(X[train_idx], y[train_idx], X[test_idx], seed=seed)
        fold_rows.append(
            {
                "fold": fold,
                "train_groups": len(np.unique(groups[train_idx])),
                "test_groups": len(np.unique(groups[test_idx])),
                **classification_metrics(y[test_idx], p),
            }
        )
    grouped = pd.DataFrame(fold_rows)

    naive_train, naive_test = naive_split(y, seed=seed)
    naive_probability = fit_probability(
        X[naive_train], y[naive_train], X[naive_test], seed=seed
    )
    naive = classification_metrics(y[naive_test], naive_probability)

    domain_rows = []
    if domains is not None:
        domains = np.asarray(domains)
        if len(domains) != len(y):
            raise ValueError("domains must match y length")
        for domain, train_idx, test_idx in leave_one_domain_out(domains):
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                continue
            p = fit_probability(X[train_idx], y[train_idx], X[test_idx], seed=seed)
            domain_rows.append(
                {
                    "held_out_domain": str(domain),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    **classification_metrics(y[test_idx], p),
                }
            )
    domain_shift = pd.DataFrame(domain_rows)

    grouped_auc = float(grouped["roc_auc"].mean())
    summary = {
        "n_samples": float(len(y)),
        "n_groups": float(len(np.unique(groups))),
        "n_features": float(X.shape[1]),
        "grouped_roc_auc_mean": grouped_auc,
        "grouped_balanced_accuracy_mean": float(grouped["balanced_accuracy"].mean()),
        "grouped_brier_mean": float(grouped["brier"].mean()),
        "naive_roc_auc": float(naive["roc_auc"]),
        "naive_balanced_accuracy": float(naive["balanced_accuracy"]),
        "naive_brier": float(naive["brier"]),
        "leakage_gap_roc_auc": float(naive["roc_auc"] - grouped_auc),
    }
    if not domain_shift.empty:
        summary["domain_shift_roc_auc_mean"] = float(domain_shift["roc_auc"].mean())
        summary["domain_shift_roc_auc_worst"] = float(domain_shift["roc_auc"].min())

    return BenchmarkReport(
        grouped_folds=grouped,
        domain_shift=domain_shift,
        summary=summary,
    )
