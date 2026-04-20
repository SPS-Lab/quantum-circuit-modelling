"""Run fixed-flux Duffing truncation benchmark with parameters loaded from /params."""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from comparison.truncation import run_truncation_benchmark
from plots.truncation import plot_truncation_benchmark
from study_config import load_study_config



def main() -> None:
    repo_root = _REPO_ROOT
    config = load_study_config(
        repo_root / "params" / "system_params.json",
        repo_root / "params" / "static_benchmark_params.json",
    )
    trunc_cfg = config.truncation_benchmark

    result = run_truncation_benchmark(
        config,
        duffing_ncut_values=list(trunc_cfg.duffing_ncut_values),
        fixed_flux=float(trunc_cfg.fixed_flux),
        duffing_truncated_dim=int(trunc_cfg.duffing_truncated_dim),
        lowest_excited_levels_to_report=int(trunc_cfg.lowest_excited_levels_to_plot),
        circuit_reference_ncut=int(trunc_cfg.circuit_reference_ncut),
        duffing_calibration_mode=str(trunc_cfg.duffing_calibration_mode),
    )

    configured_figure = Path(trunc_cfg.outputs.figure)
    if configured_figure.is_absolute():
        figure_path = configured_figure
    else:
        figure_path = repo_root / configured_figure
    title = (
        "Fixed-flux truncation benchmark: Duffing vs circuit "
        f"(flux={result.flux:.6f}, target={result.sweep_target})"
    )
    plot_truncation_benchmark(
        result,
        figure_path,
        title,
        lowest_excited_levels_to_plot=int(trunc_cfg.lowest_excited_levels_to_plot),
    )

    print("Truncation benchmark summary:")
    for key, value in result.summary.items():
        print(f"  {key}: {value:.6e}")

    print("Per-ncut metrics (GHz):")
    for ncut, trunc_dim, j, zeta in zip(
        result.duffing_ncut_values,
        result.duffing_effective_truncated_dim_values,
        result.duffing_j,
        result.duffing_zeta,
    ):
        print(
            f"  ncut={int(ncut):4d}, trunc_dim={int(trunc_dim):3d}: "
            f"J={j:.6e}, zeta={zeta:.6e}"
        )
    print(
        "Circuit reference (GHz): "
        f"ncut={result.circuit_reference_ncut}, "
        f"J={result.circuit_j:.6e}, zeta={result.circuit_zeta:.6e}"
    )
    print(
        "Duffing - circuit at max Duffing ncut "
        f"(ncut={result.max_duffing_ncut}) for reported excited levels (GHz):"
    )
    for level, diff, rel_pct in zip(
        result.max_ncut_reported_excited_levels,
        result.duffing_minus_circuit_at_max_ncut,
        result.duffing_minus_circuit_percent_of_circuit_at_max_ncut,
    ):
        rel_text = f"{float(rel_pct):.6f}%"
        if not (float(rel_pct) == float(rel_pct)):  # NaN check
            rel_text = "nan%"
        print(f"  E{int(level)}: {float(diff):.12e} ({rel_text} of circuit)")
    print(f"Wrote figure: {figure_path}")


if __name__ == "__main__":
    main()
