"""Plotting entry points for study benchmarks."""

from plotting.cz import plot_cz_benchmark
from plotting.leakage_flow import plot_leakage_flow_benchmark
from plotting.runtime import plot_runtime_benchmark
from plotting.rx import plot_rx_diagnostics_benchmark, plot_rx_populations_benchmark
from plotting.static import plot_static_benchmark, plot_static_raw_energies, plot_static_single_excitation_overlaps
from plotting.style import DEFAULT_PLOT_FONT_SIZE
from plotting.truncation import plot_circuit_truncation_benchmark, plot_duffing_truncation_benchmark

__all__ = [
    "DEFAULT_PLOT_FONT_SIZE",
    "plot_static_benchmark",
    "plot_static_raw_energies",
    "plot_static_single_excitation_overlaps",
    "plot_cz_benchmark",
    "plot_runtime_benchmark",
    "plot_rx_populations_benchmark",
    "plot_rx_diagnostics_benchmark",
    "plot_leakage_flow_benchmark",
    "plot_circuit_truncation_benchmark",
    "plot_duffing_truncation_benchmark",
]
