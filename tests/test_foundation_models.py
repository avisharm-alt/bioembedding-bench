from bioembedding_bench.diagnostics import embedding_diagnostics
from bioembedding_bench.probe_suite import compare_probes
from bioembedding_bench.synthetic import make_biological_embeddings


def test_probe_suite_compares_linear_and_mlp():
    X, y, donors, _ = make_biological_embeddings(
        n_donors=24,
        samples_per_donor=4,
        n_features=16,
        n_sites=3,
        seed=31,
    )
    result = compare_probes(X, y, donors, n_splits=3, seed=31)
    assert set(result["probe"]) == {"linear", "mlp"}
    assert len(result) == 6


def test_embedding_diagnostics():
    X, _, _, _ = make_biological_embeddings(
        n_donors=16,
        samples_per_donor=3,
        n_features=12,
        n_sites=2,
        seed=32,
    )
    d = embedding_diagnostics(X, seed=32)
    assert d["effective_rank"] > 0
    assert d["mean_l2_norm"] > 0
    assert 0 <= d["cosine_anisotropy"] <= 1
