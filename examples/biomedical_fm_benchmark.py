from bioembedding_bench import (
    benchmark_embeddings,
    compare_probes,
    embedding_diagnostics,
    make_biological_embeddings,
)

X, y, donors, sites = make_biological_embeddings(
    n_donors=80,
    samples_per_donor=8,
    n_features=128,
    n_sites=4,
    seed=17,
)

print(embedding_diagnostics(X, seed=17))
print(compare_probes(X, y, donors, n_splits=5, seed=17))
print(benchmark_embeddings(X, y, donors, domains=sites, seed=17).summary)
