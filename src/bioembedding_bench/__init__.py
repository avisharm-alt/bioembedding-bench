"""Benchmarking tools for biomedical foundation-model embeddings."""

from .diagnostics import embedding_diagnostics
from .evaluate import BenchmarkReport, benchmark_embeddings
from .probe_suite import compare_probes
from .synthetic import make_biological_embeddings

__all__ = [
    "BenchmarkReport",
    "benchmark_embeddings",
    "compare_probes",
    "embedding_diagnostics",
    "make_biological_embeddings",
]
__version__ = "0.2.0"
