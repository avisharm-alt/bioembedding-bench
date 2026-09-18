from __future__ import annotations

import numpy as np


def effective_rank(X) -> float:
    """Entropy-based effective rank of the centered embedding matrix."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    power = s**2
    if power.sum() == 0:
        return 0.0
    p = power / power.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def cosine_anisotropy(X, max_pairs: int = 5000, seed: int = 0) -> float:
    """Estimate mean absolute cosine similarity between embedding pairs."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if len(X) < 2:
        return 0.0

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    unit = X / np.maximum(norms, 1e-12)
    rng = np.random.default_rng(seed)

    n_possible = len(X) * (len(X) - 1) // 2
    n_pairs = min(max_pairs, n_possible)
    similarities = np.empty(n_pairs, dtype=float)
    for i in range(n_pairs):
        a, b = rng.choice(len(X), size=2, replace=False)
        similarities[i] = abs(float(unit[a] @ unit[b]))
    return float(similarities.mean())


def embedding_diagnostics(X, seed: int = 0) -> dict[str, float]:
    """Summarize embedding geometry for collapse/anisotropy auditing."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    norms = np.linalg.norm(X, axis=1)
    return {
        "n_samples": float(X.shape[0]),
        "n_features": float(X.shape[1]),
        "effective_rank": effective_rank(X),
        "mean_l2_norm": float(norms.mean()),
        "std_l2_norm": float(norms.std()),
        "cosine_anisotropy": cosine_anisotropy(X, seed=seed),
    }
