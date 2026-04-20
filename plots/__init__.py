"""Plotting entry points for study benchmarks."""

from plots.cz import plot_cz_benchmark
from plots.leakage import plot_leakage_benchmark
from plots.static import plot_static_benchmark
from plots.style import DEFAULT_PLOT_FONT_SIZE
from plots.truncation import plot_truncation_benchmark

__all__ = [
    "DEFAULT_PLOT_FONT_SIZE",
    "plot_static_benchmark",
    "plot_cz_benchmark",
    "plot_leakage_benchmark",
    "plot_truncation_benchmark",
]
