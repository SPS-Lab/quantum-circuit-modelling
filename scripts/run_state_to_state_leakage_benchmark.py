"""Run state-to-state leakage-current benchmark with parameters loaded from /params."""

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
from comparison.state_to_state_leakage import (
    StateToStateLeakageBenchmarkResult,
    run_state_to_state_leakage_benchmark,
)
from plotting.state_to_state_leakage import plot_state_to_state_leakage_benchmark
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


def _time_integrated_transition_currents(
    transitions: dict[str, np.ndarray],
    times_ns: np.ndarray,
) -> list[tuple[str, float]]:
    integrated: list[tuple[str, float]] = []
    for label, trace in transitions.items():
        area = _time_integral(np.clip(np.asarray(trace, dtype=float), 0.0, None), times_ns)
        integrated.append((str(label), float(area)))
    return sorted(integrated, key=lambda x: x[1], reverse=True)


def _print_top_transitions(
    *,
    model_name: str,
    transitions: dict[str, np.ndarray],
    times_ns: np.ndarray,
    top_k: int,
) -> None:
    if not transitions:
        print(f"Top state-to-state leakage currents ({model_name}): no transition traces available")
        return
    k = int(max(1, top_k))
    ranked = _time_integrated_transition_currents(transitions, times_ns)
    top = ranked[:k]
    total = float(sum(val for _, val in ranked))
    top_total = float(sum(val for _, val in top))

    print(f"Top state-to-state leakage currents ({model_name}, integrated over time):")
    for label, value in top:
        share = 0.0 if total <= 0.0 else value / total
        print(f"  {label}: {value:.6e} ({100.0 * share:.2f}% of total comp->leak current)")
    print(
        f"Top-{k} cumulative integrated comp->leak current ({model_name}): "
        f"{top_total:.6e} ({100.0 * (0.0 if total <= 0.0 else top_total / total):.2f}% of total)"
    )


def main() -> None:
    args = _parse_args()
    repo_root = _REPO_ROOT
    config = load_study_config(
        repo_root / "params" / "system_params.json",
        repo_root / "params" / "static_benchmark_params.json",
    )

    s2s_cfg = config.state_to_state_leakage_benchmark
    target_total_time_ns = float(s2s_cfg.total_time_ns)
    ramp_time_ns = float(s2s_cfg.ramp_time_ns)
    hold_time_ns = target_total_time_ns - 2.0 * ramp_time_ns
    dt_ns = float(s2s_cfg.dt_ns)
    top_transition_rows = int(s2s_cfg.top_transition_rows)

    static_figure = repo_root / config.static_benchmark.outputs.figure
    figure_path = static_figure.with_name("model_comparison_state_to_state_leakage.pdf")
    results_path = (
        _resolve_repo_relative(repo_root, args.results)
        if args.results is not None
        else default_results_path_for_figure(figure_path)
    )

    if args.plot_only:
        result = load_result_hdf5(
            results_path,
            StateToStateLeakageBenchmarkResult,
            expected_benchmark_name="state_to_state_leakage",
        )
    else:
        result = run_state_to_state_leakage_benchmark(
            config,
            ramp_time_ns=ramp_time_ns,
            hold_time_ns=hold_time_ns,
            dt_ns=dt_ns,
        )
        save_result_hdf5(result, results_path, benchmark_name="state_to_state_leakage")

    title = "State-to-state leakage currents from |11>: computational -> leakage"
    plot_state_to_state_leakage_benchmark(
        result,
        figure_path,
        title,
        top_transition_rows=top_transition_rows,
    )

    print("State-to-state leakage benchmark summary:")
    for key, value in result.summary.items():
        print(f"  {key}: {value:.6e}")

    if result.duffing_max_transition_label_11:
        print(f"Max integrated transition (duffing): {result.duffing_max_transition_label_11}")
    if result.circuit_max_transition_label_11:
        print(f"Max integrated transition (circuit): {result.circuit_max_transition_label_11}")

    _print_top_transitions(
        model_name="duffing",
        transitions=result.duffing_comp_to_leak_currents_11,
        times_ns=result.times_ns,
        top_k=top_transition_rows,
    )
    _print_top_transitions(
        model_name="circuit",
        transitions=result.circuit_comp_to_leak_currents_11,
        times_ns=result.times_ns,
        top_k=top_transition_rows,
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
    main()
