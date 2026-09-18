"""Evaluation tools for frozen biological embeddings."""

from .evaluate import BenchmarkReport, benchmark_embeddings
from .synthetic import make_biological_embeddings

__all__ = ["BenchmarkReport", "benchmark_embeddings", "make_biological_embeddings"]
__version__ = "0.1.0"
