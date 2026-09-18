from __future__ import annotations

import numpy as np

from .evaluate import benchmark_embeddings


def group_label_permutation(
    X,
    y,
    groups,
    domains=None,
    n_permutations: int = 100,
    n_splits: int = 5,
    seed: int = 0,
) -> dict[str, object]:
    """Shuffle labels between groups while preserving within-group dependence."""
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)

    labels = []
    for group in unique_groups:
        group_labels = np.unique(y[groups == group])
        if len(group_labels) != 1:
            raise ValueError("each group must have one label for permutation testing")
        labels.append(group_labels[0])
    labels = np.asarray(labels)

    observed = benchmark_embeddings(
        X, y, groups, domains=domains, n_splits=n_splits, seed=seed
    ).summary["grouped_roc_auc_mean"]

    rng = np.random.default_rng(seed)
    scores = []
    for i in range(n_permutations):
        shuffled = rng.permutation(labels)
        lookup = dict(zip(unique_groups, shuffled))
        y_perm = np.asarray([lookup[g] for g in groups], dtype=int)
        report = benchmark_embeddings(
            X,
            y_perm,
            groups,
            domains=domains,
            n_splits=n_splits,
            seed=seed + i + 1,
        )
        scores.append(report.summary["grouped_roc_auc_mean"])

    scores = np.asarray(scores, dtype=float)
    p = (1 + np.sum(scores >= observed)) / (n_permutations + 1)
    return {
        "observed": float(observed),
        "null_mean": float(scores.mean()),
        "p_value": float(p),
        "null_scores": scores.tolist(),
    }
