# bioembedding-bench

**Benchmarking embeddings from biomedical foundation models.**

`bioembedding-bench` is an AI/ML evaluation suite for frozen representations produced by biomedical foundation models across MRI, EEG, omics, pathology, and multimodal clinical data.

The main question is: **does a pretrained model learn transferable biological signal, or are its embeddings dominated by donor identity, site, scanner, batch, or other nuisance structure?**

## AI/ML focus

- **Foundation-model embedding evaluation**
- **Linear vs nonlinear probing**
- **Donor-held-out generalization**
- **Naive-vs-grouped leakage-gap analysis**
- **Leave-one-site-out domain shift**
- **Representation-collapse diagnostics**
- **Calibration with Brier score**
- **Group-level permutation nulls**
- **Synthetic stress tests with controllable donor/site confounding**

## Typical workflow

1. Run a pretrained biomedical model on your dataset.
2. Export one embedding vector per sample.
3. Provide labels plus donor/subject IDs and optional site/scanner metadata.
4. Benchmark how much downstream signal transfers under realistic held-out conditions.

## Install

```bash
git clone https://github.com/avisharm-alt/bioembedding-bench
cd bioembedding-bench
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quickstart

```python
from bioembedding_bench.synthetic import make_biological_embeddings
from bioembedding_bench.evaluate import benchmark_embeddings
from bioembedding_bench.probe_suite import compare_probes
from bioembedding_bench.diagnostics import embedding_diagnostics

X, y, donors, sites = make_biological_embeddings(seed=17)

report = benchmark_embeddings(
    X, y, donors, domains=sites, seed=17
)
print(report.summary)

probe_results = compare_probes(
    X, y, donors, n_splits=5, seed=17
)
print(probe_results)

print(embedding_diagnostics(X))
```

## Probe suite

Two downstream models are intentionally compared:

- **Linear probe:** asks whether target information is directly accessible in embedding space.
- **MLP probe:** asks whether useful signal is present but organized nonlinearly.

A large MLP-over-linear gain can indicate nonlinear structure. A large naive-over-donor-held-out gain can indicate donor-specific leakage. A large held-out-site drop can indicate domain dependence.

## Representation diagnostics

The benchmark reports:

- Effective rank
- Mean embedding norm
- Norm dispersion
- Cosine anisotropy

These are useful for detecting low-rank collapse or highly anisotropic representations before downstream model fitting.

## CLI

Synthetic benchmark:

```bash
bioembedding-bench demo \
  --donors 80 \
  --samples-per-donor 8 \
  --features 128 \
  --sites 4 \
  --seed 17 \
  --out results/demo
```

Evaluate exported embeddings:

```bash
bioembedding-bench evaluate embeddings.csv \
  --label disease \
  --group donor_id \
  --site site \
  --feature-prefix emb_ \
  --out results/model
```

## Intended model classes

This benchmark is suitable for embeddings from:

- MRI foundation models
- EEG/self-supervised signal encoders
- vision transformers for pathology
- multimodal medical encoders
- protein/omics foundation models
- clinical representation-learning systems

## Interpretation

This repo evaluates **representations**, not clinical utility. High downstream performance does not establish clinical validity; the benchmark is intended to make representation claims more rigorous and harder to inflate through leakage or domain shortcuts.

## License

MIT
