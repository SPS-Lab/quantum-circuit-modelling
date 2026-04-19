"""Benchmark runners for the paper study."""

from study.comparison.cz import run_cz_benchmark
from study.comparison.leakage import run_leakage_benchmark
from study.comparison.static import StaticBenchmarkResult, run_static_benchmark

__all__ = [
    "StaticBenchmarkResult",
    "run_static_benchmark",
    "run_cz_benchmark",
    "run_leakage_benchmark",
]
