from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit


def grouped_splits(y, groups, n_splits: int = 5, seed: int = 0):
    y = np.asarray(y)
    groups = np.asarray(groups)
    if len(np.unique(groups)) < n_splits:
        raise ValueError("not enough unique groups for requested n_splits")
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    yield from splitter.split(np.zeros((len(y), 1)), y, groups)


def naive_split(y, test_size: float = 0.2, seed: int = 0):
    y = np.asarray(y)
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    return next(splitter.split(np.zeros((len(y), 1)), y))


def leave_one_domain_out(domains):
    domains = np.asarray(domains)
    for domain in np.unique(domains):
        test = np.flatnonzero(domains == domain)
        train = np.flatnonzero(domains != domain)
        yield domain, train, test
