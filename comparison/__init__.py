"""Benchmark runners for the paper study."""

from comparison.cz import CzBenchmarkResult, run_cz_benchmark
from comparison.leakage import LeakageBenchmarkResult, run_leakage_benchmark
from comparison.leakage_flow import LeakageFlowBenchmarkResult, run_leakage_flow_benchmark
from comparison.regime_map import compare_model1_model2_against_scqubits
from comparison.state_to_state_leakage import (
    StateToStateLeakageBenchmarkResult,
    run_state_to_state_leakage_benchmark,
)
from comparison.static import StaticBenchmarkResult, run_static_benchmark
from comparison.truncation import TruncationBenchmarkResult, run_truncation_benchmark

__all__ = [
    "StaticBenchmarkResult",
    "CzBenchmarkResult",
    "LeakageBenchmarkResult",
    "LeakageFlowBenchmarkResult",
    "StateToStateLeakageBenchmarkResult",
    "TruncationBenchmarkResult",
    "run_static_benchmark",
    "run_cz_benchmark",
    "run_leakage_benchmark",
    "run_leakage_flow_benchmark",
    "run_state_to_state_leakage_benchmark",
    "run_truncation_benchmark",
    "compare_model1_model2_against_scqubits",
]
