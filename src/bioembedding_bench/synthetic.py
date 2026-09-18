from __future__ import annotations

import numpy as np


def make_biological_embeddings(
    n_donors: int = 80,
    samples_per_donor: int = 8,
    n_features: int = 128,
    n_sites: int = 4,
    biological_signal: float = 0.7,
    donor_signal: float = 1.3,
    site_signal: float = 0.7,
    noise: float = 1.0,
    seed: int = 0,
):
    """Generate frozen embeddings with donor and site nuisance structure."""
    if n_donors < 4:
        raise ValueError("n_donors must be >= 4")
    if n_sites < 2:
        raise ValueError("n_sites must be >= 2")

    rng = np.random.default_rng(seed)
    labels = np.arange(n_donors) % 2
    rng.shuffle(labels)

    donor_vectors = rng.normal(0, donor_signal, size=(n_donors, n_features))
    site_vectors = rng.normal(0, site_signal, size=(n_sites, n_features))
    biology = rng.normal(size=n_features)
    biology /= np.linalg.norm(biology)

    X, y, donors, sites = [], [], [], []
    for donor in range(n_donors):
        site = donor % n_sites
        label = int(labels[donor])
        class_vector = (2 * label - 1) * biological_signal * biology

        for _ in range(samples_per_donor):
            embedding = (
                donor_vectors[donor]
                + site_vectors[site]
                + class_vector
                + rng.normal(0, noise, size=n_features)
            )
            X.append(embedding)
            y.append(label)
            donors.append(f"d{donor:03d}")
            sites.append(f"site_{site}")

    return (
        np.asarray(X),
        np.asarray(y, dtype=int),
        np.asarray(donors),
        np.asarray(sites),
    )
