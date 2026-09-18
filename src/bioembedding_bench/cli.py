from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .evaluate import benchmark_embeddings
from .nulls import group_label_permutation
from .synthetic import make_biological_embeddings


def _write(report, out: Path, null=None):
    out.mkdir(parents=True, exist_ok=True)
    report.grouped_folds.to_csv(out / "grouped_folds.csv", index=False)
    report.domain_shift.to_csv(out / "domain_shift.csv", index=False)

    summary = dict(report.summary)
    if null is not None:
        summary["permutation_test"] = {
            k: v for k, v in null.items() if k != "null_scores"
        }
        pd.DataFrame({"null_roc_auc": null["null_scores"]}).to_csv(
            out / "null_distribution.csv", index=False
        )

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parser():
    p = argparse.ArgumentParser(prog="bioembedding-bench")
    sub = p.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo")
    demo.add_argument("--donors", type=int, default=80)
    demo.add_argument("--samples-per-donor", type=int, default=8)
    demo.add_argument("--features", type=int, default=128)
    demo.add_argument("--sites", type=int, default=4)
    demo.add_argument("--splits", type=int, default=5)
    demo.add_argument("--permutations", type=int, default=25)
    demo.add_argument("--seed", type=int, default=17)
    demo.add_argument("--out", type=Path, default=Path("results/demo"))

    ev = sub.add_parser("evaluate")
    ev.add_argument("csv", type=Path)
    ev.add_argument("--label", required=True)
    ev.add_argument("--group", required=True)
    ev.add_argument("--site")
    ev.add_argument("--feature-prefix", default="emb_")
    ev.add_argument("--splits", type=int, default=5)
    ev.add_argument("--seed", type=int, default=17)
    ev.add_argument("--out", type=Path, default=Path("results/eval"))
    return p


def main():
    args = parser().parse_args()

    if args.command == "demo":
        X, y, groups, sites = make_biological_embeddings(
            n_donors=args.donors,
            samples_per_donor=args.samples_per_donor,
            n_features=args.features,
            n_sites=args.sites,
            seed=args.seed,
        )
        report = benchmark_embeddings(
            X, y, groups, domains=sites, n_splits=args.splits, seed=args.seed
        )
        null = group_label_permutation(
            X,
            y,
            groups,
            domains=sites,
            n_permutations=args.permutations,
            n_splits=args.splits,
            seed=args.seed,
        )
        _write(report, args.out, null)
        return

    df = pd.read_csv(args.csv)
    features = [c for c in df.columns if c.startswith(args.feature_prefix)]
    if not features:
        raise ValueError("no feature columns matched --feature-prefix")

    domains = df[args.site].to_numpy() if args.site else None
    report = benchmark_embeddings(
        df[features].to_numpy(),
        df[args.label].to_numpy(),
        df[args.group].to_numpy(),
        domains=domains,
        n_splits=args.splits,
        seed=args.seed,
    )
    _write(report, args.out)


if __name__ == "__main__":
    main()
