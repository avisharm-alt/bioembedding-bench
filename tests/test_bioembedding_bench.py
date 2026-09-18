from bioembedding_bench.evaluate import benchmark_embeddings
from bioembedding_bench.synthetic import make_biological_embeddings


def test_benchmark_runs_with_group_and_domain_shift():
    X, y, donors, sites = make_biological_embeddings(
        n_donors=24,
        samples_per_donor=4,
        n_features=16,
        n_sites=3,
        seed=5,
    )
    report = benchmark_embeddings(
        X, y, donors, domains=sites, n_splits=3, seed=5
    )
    assert len(report.grouped_folds) == 3
    assert len(report.domain_shift) == 3
    assert 0.0 <= report.summary["grouped_roc_auc_mean"] <= 1.0
    assert "leakage_gap_roc_auc" in report.summary


def test_synthetic_generation_is_reproducible():
    a = make_biological_embeddings(
        n_donors=12, samples_per_donor=2, n_features=8, n_sites=2, seed=9
    )
    b = make_biological_embeddings(
        n_donors=12, samples_per_donor=2, n_features=8, n_sites=2, seed=9
    )
    for x, y in zip(a, b):
        assert (x == y).all()
