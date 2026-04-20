"""Benchmark runners for the paper study."""

from comparison.cz import CzBenchmarkResult, run_cz_benchmark
from comparison.leakage import run_leakage_benchmark
from comparison.regime_map import compare_model1_model2_against_scqubits
from comparison.static import StaticBenchmarkResult, run_static_benchmark

__all__ = [
    "StaticBenchmarkResult",
    "CzBenchmarkResult",
    "run_static_benchmark",
    "run_cz_benchmark",
    "run_leakage_benchmark",
    "compare_model1_model2_against_scqubits",
]
