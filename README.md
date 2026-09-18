# bioembedding-bench

A compact benchmark for testing whether biological embeddings carry transferable signal rather than donor, batch, or site identity.

The central question is: **when an encoder produces an embedding for biological or clinical data, does a simple downstream probe generalize across people and acquisition domains?**

This project is built for frozen representations from MRI, EEG, omics, pathology, and other biomedical encoders.

## Benchmark axes

- **Group generalization** — donors/subjects are held out across folds
- **Leakage gap** — compares naive row-wise splitting with donor-held-out evaluation
- **Domain shift** — leave-one-site-out evaluation
- **Linear probe performance** — standardized logistic regression only
- **Calibration** — ROC AUC, balanced accuracy, and Brier score
- **Null testing** — subject-level label permutation
- **Synthetic stress test** — controllable biological signal, donor fingerprint, and site shift

## Install

```bash
git clone https://github.com/avisharm-alt/bioembedding-bench
cd bioembedding-bench
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quickstart

Run the synthetic stress test:

```bash
bioembedding-bench demo \
  --donors 80 \
  --samples-per-donor 8 \
  --features 128 \
  --sites 4 \
  --seed 17 \
  --out results/demo
```

Benchmark your own embeddings:

```bash
bioembedding-bench evaluate embeddings.csv \
  --label disease \
  --group donor_id \
  --site site \
  --feature-prefix emb_ \
  --out results/model
```

## Input contract

Each row is a sample. Multiple rows may belong to one donor.

```text
donor_id,site,disease,emb_0,emb_1,...,emb_511
d001,A,0,...
d001,A,0,...
d002,B,1,...
```

The donor and site columns are metadata only and are excluded from the probe features.

## Output

- `grouped_folds.csv` — donor-held-out cross-validation results
- `domain_shift.csv` — leave-one-site-out results
- `summary.json` — aggregate performance and leakage-gap audit
- `null_distribution.csv` — optional subject-level permutation null

## Python API

```python
from bioembedding_bench.synthetic import make_biological_embeddings
from bioembedding_bench.evaluate import benchmark_embeddings

X, y, donors, sites = make_biological_embeddings(seed=17)
report = benchmark_embeddings(X, y, donors, sites, seed=17)

print(report.summary)
print(report.grouped_folds)
print(report.domain_shift)
```

## Interpretation

A large **naive-minus-grouped leakage gap** suggests that row-wise validation is exploiting repeated-donor structure. A large drop on a held-out site suggests domain-specific features. Neither automatically invalidates an encoder; both identify what claim the representation can and cannot currently support.

## Scope

This repository evaluates representations. It does not train foundation models and does not claim clinical validity. The goal is to make downstream evaluation harder to fool.

## License

MIT
