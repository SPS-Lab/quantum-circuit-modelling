"""Run leakage benchmark with parameters loaded from /params."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark_results_io import (
    default_results_path_for_figure,
    load_result_hdf5,
    save_result_hdf5,
)
from comparison.leakage import LeakageBenchmarkResult, run_leakage_benchmark
from plotting.leakage import plot_leakage_benchmark
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


def _time_integral(values: np.ndarray, times_ns: np.ndarray) -> float:
    y = np.asarray(values, dtype=float).ravel()
    t = np.asarray(times_ns, dtype=float).ravel()
    if y.shape != t.shape:
        raise ValueError("values and times_ns must have matching shape")
    try:
        return float(np.trapezoid(y, x=t))
    except AttributeError:  # pragma: no cover - compatibility fallback
        return float(np.trapz(y, x=t))


def _time_integrated_leakage_share(state_population: np.ndarray, leakage: np.ndarray, times_ns: np.ndarray) -> float:
    state_area = _time_integral(np.clip(np.asarray(state_population, dtype=float), 0.0, None), times_ns)
    leak_area = _time_integral(np.clip(np.asarray(leakage, dtype=float), 0.0, None), times_ns)
    return 0.0 if leak_area <= 0.0 else float(state_area / leak_area)


def _time_integrated_destination_shares(
    destinations: dict[str, np.ndarray],
    leakage: np.ndarray,
    times_ns: np.ndarray,
) -> list[tuple[str, float]]:
    leak_area = _time_integral(np.clip(np.asarray(leakage, dtype=float), 0.0, None), times_ns)
    if leak_area <= 0.0:
        return [(str(label), 0.0) for label in sorted(destinations.keys())]

    shares: list[tuple[str, float]] = []
    for label, trace in destinations.items():
        share = _time_integrated_leakage_share(trace, leakage, times_ns)
        shares.append((str(label), float(share)))
    return sorted(shares, key=lambda x: x[1], reverse=True)


def _print_destination_shares(
    *,
    model_name: str,
    destinations: dict[str, np.ndarray],
    leakage: np.ndarray,
    times_ns: np.ndarray,
    min_share_to_print: float = 1e-8,
    top_k_sum: int = 5,
) -> None:
    if not destinations:
        print(f"Time-integrated leakage shares by destination ({model_name}): no destination traces available")
        return
    min_share = float(max(0.0, min_share_to_print))
    shares = _time_integrated_destination_shares(destinations, leakage, times_ns)
    shown = [(label, share) for (label, share) in shares if share >= min_share]
    hidden_count = len(shares) - len(shown)
    k = int(max(1, top_k_sum))
    top = shares[:k]
    top_sum = float(sum(share for _, share in top))
    top_labels = ", ".join(label for label, _ in top)
    print(f"Time-integrated leakage shares by destination ({model_name}):")
    for label, share in shown:
        print(f"  {label}: {share:.6e}")
    if hidden_count > 0:
        print(f"  ... ({hidden_count} destinations below {min_share:.1e} omitted)")
    print(
        f"Top-{k} cumulative time-integrated leakage share ({model_name}): "
        f"{top_sum:.6e} ({100.0 * top_sum:.2f}%)"
    )
    print(f"Top-{k} states ({model_name}): {top_labels}")


def main() -> None:
    args = _parse_args()
    repo_root = _REPO_ROOT
    config = load_study_config(
        repo_root / "params" / "system_params.json",
        repo_root / "params" / "benchmark_params.json",
    )
    leakage_cfg = config.leakage_benchmark
    target_total_time_ns = float(leakage_cfg.total_time_ns)
    ramp_time_ns = float(leakage_cfg.ramp_time_ns)
    hold_time_ns = target_total_time_ns - 2.0 * ramp_time_ns
    dt_ns = float(leakage_cfg.dt_ns)
    top_destination_rows = int(leakage_cfg.top_destination_rows)

    static_figure = repo_root / config.static_benchmark.outputs.figure
    figure_path = static_figure.with_name("model_comparison_leakage.pdf")
    results_path = (
        _resolve_repo_relative(repo_root, args.results)
        if args.results is not None
        else default_results_path_for_figure(figure_path)
    )

    if args.plot_only:
        result = load_result_hdf5(
            results_path,
            LeakageBenchmarkResult,
            expected_benchmark_name="leakage",
        )
    else:
        result = run_leakage_benchmark(
            config,
            ramp_time_ns=ramp_time_ns,
            hold_time_ns=hold_time_ns,
            dt_ns=dt_ns,
        )
        save_result_hdf5(result, results_path, benchmark_name="leakage")

    title = "Leakage benchmark from |11>: effective vs Duffing vs circuit"
    plot_leakage_benchmark(result, figure_path, title, top_destination_rows=top_destination_rows)

    print("Leakage benchmark summary:")
    for key, value in result.summary.items():
        print(f"  {key}: {value:.6e}")
    duf_frac_110 = result.summary.get("duffing_fraction_of_time_integrated_leakage_to_state_110_11")
    cir_frac_110 = result.summary.get("circuit_fraction_of_time_integrated_leakage_to_state_110_11")
    if duf_frac_110 is None:
        duf_frac_110 = _time_integrated_leakage_share(
            state_population=result.duffing_state_110_11,
            leakage=result.duffing_leakage_11,
            times_ns=result.times_ns,
        )
    if cir_frac_110 is None:
        cir_frac_110 = _time_integrated_leakage_share(
            state_population=result.circuit_state_110_11,
            leakage=result.circuit_leakage_11,
            times_ns=result.times_ns,
        )
    print(
        "Time-integrated leakage share into |1,1,0>: "
        f"duffing={duf_frac_110:.6e}, circuit={cir_frac_110:.6e}"
    )
    print(
        "Tracked leakage destinations from |11>: "
        f"duffing={len(result.duffing_leakage_destination_populations_11)}, "
        f"circuit={len(result.circuit_leakage_destination_populations_11)}"
    )
    _print_destination_shares(
        model_name="duffing",
        destinations=result.duffing_leakage_destination_populations_11,
        leakage=result.duffing_leakage_11,
        times_ns=result.times_ns,
    )
    _print_destination_shares(
        model_name="circuit",
        destinations=result.circuit_leakage_destination_populations_11,
        leakage=result.circuit_leakage_11,
        times_ns=result.times_ns,
    )
    print(
        "Selected pulse: "
        f"sweep_target={result.sweep_target}, idle_flux={result.idle_flux:.6f}, "
        f"target_flux={result.target_flux:.6f}, ramp_time_ns={result.ramp_time_ns:.3f}, "
        f"hold_time_ns={result.hold_time_ns:.3f}, dt_ns={result.dt_ns:.3f}"
    )
    if args.plot_only:
        print(f"Loaded results: {results_path}")
    else:
        print(f"Wrote results: {results_path}")
    print(f"Wrote figure: {figure_path}")


if __name__ == "__main__":
    run_main_with_timing(main)
