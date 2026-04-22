"""Run CZ benchmark with parameters loaded from /params."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark_results_io import (
    default_results_path_for_figure,
    load_result_hdf5,
    save_result_hdf5,
)
from comparison.cz import CzBenchmarkResult, run_cz_benchmark
from plotting.cz import plot_cz_benchmark
from runtime_utils import run_main_with_timing
from study_config import load_study_config



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Path to HDF5 results file (default: figure path with .h5 suffix).",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip benchmark computation and plot from an existing HDF5 results file.",
    )
    return parser.parse_args()


def _resolve_repo_relative(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path)


def main() -> None:
    args = _parse_args()
    repo_root = _REPO_ROOT
    config = load_study_config(
        repo_root / "params" / "system_params.json",
        repo_root / "params" / "benchmark_params.json",
    )
    cz_cfg = config.cz_benchmark
    target_total_time_ns = float(cz_cfg.total_time_ns)
    ramp_time_ns = float(cz_cfg.ramp_time_ns)
    dt_ns = float(cz_cfg.dt_ns)
    hold_time_ns = target_total_time_ns - 2.0 * ramp_time_ns
    enable_hold_time_scan = bool(cz_cfg.enable_hold_time_scan)

    static_figure = repo_root / config.static_benchmark.outputs.figure
    figure_path = static_figure.with_name("model_comparison_cz_dynamics.pdf")
    results_path = (
        _resolve_repo_relative(repo_root, args.results)
        if args.results is not None
        else default_results_path_for_figure(figure_path)
    )

    if args.plot_only:
        result = load_result_hdf5(
            results_path,
            CzBenchmarkResult,
            expected_benchmark_name="cz",
        )
    else:
        result = run_cz_benchmark(
            config,
            ramp_time_ns=ramp_time_ns,
            hold_time_ns=None if enable_hold_time_scan else hold_time_ns,
            dt_ns=dt_ns,
            enable_hold_time_scan=enable_hold_time_scan,
            scan_dt_ns=float(cz_cfg.scan_dt_ns),
            scan_max_hold_ns=float(cz_cfg.scan_max_hold_ns),
            scan_leakage_penalty=float(cz_cfg.scan_leakage_penalty),
        )
        save_result_hdf5(result, results_path, benchmark_name="cz")

    title = "CZ Benchmark: Flux And CPhase"
    plot_cz_benchmark(result, figure_path, title)

    print("CZ benchmark summary:")
    cz_summary_keys = [
        "effective_final_conditional_phase_rad",
        "duffing_final_conditional_phase_rad",
        "circuit_final_conditional_phase_rad",
        "circuit_final_phase_error_to_pi_rad",
        "effective_final_phase_error_vs_circuit_rad",
        "duffing_final_phase_error_vs_circuit_rad",
        "effective_populations_rmse_vs_circuit",
        "duffing_populations_rmse_vs_circuit",
        "ramp_time_ns",
        "hold_time_ns",
        "dt_ns",
    ]
    for key in cz_summary_keys:
        if key in result.summary:
            print(f"  {key}: {result.summary[key]:.6e}")
    print(
        "Selected pulse: "
        f"sweep_target={result.sweep_target}, idle_flux={result.idle_flux:.6f}, "
        f"target_flux={result.target_flux:.6f}, ramp_time_ns={result.ramp_time_ns:.3f}, "
        f"hold_time_ns={result.hold_time_ns:.3f}, dt_ns={result.dt_ns:.3f}"
    )
    if result.scan_hold_times_ns.size > 0:
        print("Hold scan (ns, phase_err_to_pi_rad, score):")
        for h, err, score in zip(result.scan_hold_times_ns, result.scan_phase_error_rad, result.scan_scores):
            print(f"  {h:.6f}, {err:.6e}, {score:.6e}")
    if args.plot_only:
        print(f"Loaded results: {results_path}")
    else:
        print(f"Wrote results: {results_path}")
    print(f"Wrote figure: {figure_path}")


if __name__ == "__main__":
    run_main_with_timing(main)
