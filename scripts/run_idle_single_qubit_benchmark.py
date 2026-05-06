"""Run isolated idle single-qubit comparison across circuit, Duffing, and effective models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark_cli_reporting import CliReporter
from benchmark_results_io import default_results_path_for_figure, load_result_hdf5, save_result_hdf5
from comparison.idle_single_qubit import IdleSingleQubitBenchmarkResult, run_idle_single_qubit_benchmark
from plotting.idle_single_qubit import plot_idle_single_qubit_benchmark
from study_config import load_study_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qubit",
        choices=("q1", "q2"),
        default=None,
        help="Use a physical repo qubit instead of the default toy 5 GHz isolated qubit.",
    )
    parser.add_argument(
        "--toy-w01-ghz",
        type=float,
        default=5.0,
        help="Target transition frequency for the default ideal-LC toy qubit.",
    )
    parser.add_argument(
        "--total-time-ns",
        type=float,
        default=2.0,
        help="Total idle evolution time in ns.",
    )
    parser.add_argument(
        "--dt-ns",
        type=float,
        default=0.001,
        help="Time step in ns.",
    )
    parser.add_argument(
        "--initial-state",
        choices=("plus", "ground", "excited"),
        default="plus",
        help="Initial state in the model energy basis.",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=None,
        help="Output figure path. Default: results/model_comparison_idle_single_<qubit>.pdf",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Path to HDF5 results file (default: figure path with .h5 suffix).",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip computation and replot from an existing HDF5 results file.",
    )
    return parser.parse_args()


def _resolve_repo_relative(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path)


def main() -> None:
    args = _parse_args()
    repo_root = _REPO_ROOT
    reporter = CliReporter(benchmark_name="idle_single_qubit", script_name=Path(__file__).name)
    system_params_path = repo_root / "params" / "system_params.json"
    study_params_path = repo_root / "params" / "benchmark_params.json"
    config = load_study_config(system_params_path, study_params_path)

    if args.qubit is None:
        case_label = f"{float(args.toy_w01_ghz):g}ghz"
        default_figure = repo_root / "results" / f"single-{case_label}" / f"model_comparison_idle_single_{case_label}.pdf"
    else:
        case_label = str(args.qubit)
        default_figure = repo_root / "results" / f"single-{case_label}" / f"model_comparison_idle_single_{case_label}.pdf"
    figure_path = _resolve_repo_relative(repo_root, args.figure) if args.figure is not None else default_figure
    results_path = (
        _resolve_repo_relative(repo_root, args.results)
        if args.results is not None
        else default_results_path_for_figure(figure_path)
    )

    if args.plot_only:
        result = load_result_hdf5(
            results_path,
            IdleSingleQubitBenchmarkResult,
            expected_benchmark_name="idle_single_qubit",
        )
    else:
        result = run_idle_single_qubit_benchmark(
            config,
            qubit=args.qubit,
            total_time_ns=float(args.total_time_ns),
            dt_ns=float(args.dt_ns),
            toy_w01_ghz=float(args.toy_w01_ghz),
            initial_state=str(args.initial_state),
        )
        save_result_hdf5(result, results_path, benchmark_name="idle_single_qubit")

    if result.qubit == "toy":
        title = (
            f"Idle isolated toy qubit at {float(result.summary['toy_target_w01_ghz']):.3f} GHz from {result.initial_state_label}: "
            f"circuit vs Duffing vs effective"
        )
    else:
        title = (
            f"Idle isolated {result.qubit} at flux {result.flux:.6f} from {result.initial_state_label}: "
            f"circuit vs Duffing vs effective (effective source={result.effective_source})"
        )
    plot_idle_single_qubit_benchmark(result, figure_path, title)

    reporter.line("Idle single-qubit summary:")
    reporter.line(f"  case={result.case_label}, qubit={result.qubit}, flux={result.flux:.6f}")
    reporter.line(f"  initial_state={result.initial_state_label}")
    reporter.line(f"  effective_source={result.effective_source}")
    if result.qubit == "toy":
        reporter.line(
            "  toy LC params="
            f"(target_w01={float(result.summary['toy_target_w01_ghz']):.6f} GHz, "
            f"alpha={float(result.summary['toy_alpha_ghz']):.6f} GHz)"
        )
    reporter.line(
        "  dims="
        f"(circuit={int(result.summary['circuit_dim'])}, "
        f"duffing={int(result.summary['duffing_dim'])}, "
        f"effective={int(result.summary['effective_dim'])})"
    )
    reporter.line(
        "  w01 [GHz]="
        f"(circuit={float(result.summary['circuit_w01_ghz']):.6f}, "
        f"duffing={float(result.summary['duffing_w01_ghz']):.6f}, "
        f"effective={float(result.summary['effective_w01_ghz']):.6f})"
    )
    reporter.line(
        "  alpha [GHz]="
        f"(circuit={float(result.summary['circuit_alpha_ghz']):.6f}, "
        f"duffing={float(result.summary['duffing_alpha_ghz']):.6f}, "
        f"effective={float(result.summary['effective_alpha_ghz']):.6f})"
    )
    reporter.line(
        "  w01 deltas [MHz] vs circuit="
        f"(duffing={float(result.summary['duffing_minus_circuit_w01_mhz']):.3f}, "
        f"effective={float(result.summary['effective_minus_circuit_w01_mhz']):.3f})"
    )
    reporter.line(
        "  max logical leakage="
        f"(circuit={float(result.summary['max_circuit_logical_leakage']):.3e}, "
        f"duffing={float(result.summary['max_duffing_logical_leakage']):.3e}, "
        f"effective={float(result.summary['max_effective_logical_leakage']):.3e})"
    )
    if args.plot_only:
        reporter.line(f"Loaded results: {results_path}")
    else:
        reporter.line(f"Wrote results: {results_path}")
    reporter.line(f"Wrote figure: {figure_path}")
    reporter.add_runtime_line()
    reporter.persist(results_path)


if __name__ == "__main__":
    main()
