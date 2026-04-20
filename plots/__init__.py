"""Plotting entry points for study benchmarks."""

from plots.cz import plot_cz_benchmark
from plots.leakage import plot_leakage_benchmark
from plots.static import plot_static_benchmark

__all__ = ["plot_static_benchmark", "plot_cz_benchmark", "plot_leakage_benchmark"]
