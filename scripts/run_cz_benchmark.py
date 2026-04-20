"""Run CZ benchmark with parameters loaded from /params."""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from comparison.cz import run_cz_benchmark
from plots.cz import plot_cz_benchmark
from study_config import load_study_config



def main() -> None:
    repo_root = _REPO_ROOT
    config = load_study_config(
        repo_root / "params" / "system_params.json",
        repo_root / "params" / "static_benchmark_params.json",
    )
    result = run_cz_benchmark(config)

    static_figure = repo_root / config.static_benchmark.outputs.figure
    figure_path = static_figure.with_name("model_comparison_cz_dynamics.pdf")
    title = "CZ-relevant dynamics: effective vs Duffing vs circuit"
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
    print(f"Wrote figure: {figure_path}")


if __name__ == "__main__":
    main()
