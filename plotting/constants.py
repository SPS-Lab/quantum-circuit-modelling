"""Project constants for plotting style and geometry."""

from __future__ import annotations

ACTIVE_BENCHMARK_STYLE: str = "paper"

ACM_SIGCONF_COLUMN_WIDTH_PT: float = 241.14749
ACM_SIGCONF_TEXT_WIDTH_PT: float = 506.295
TEX_POINTS_PER_INCH: float = 72.27

MODEL_ALPHA_CIRCUIT: float = 1.0
MODEL_ALPHA_DUFFING: float = 0.98
MODEL_ALPHA_EFFECTIVE: float = 0.98

MODEL_LEGEND_BBOX_TO_ANCHOR: tuple[float, float] = (0.5, 1.01)
BENCHMARK_TIGHT_LAYOUT_RECT: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.93)
BENCHMARK_TIGHT_LAYOUT_H_PAD: float = 1.2
BENCHMARK_TIGHT_LAYOUT_W_PAD: float = 0.9

TRUNCATION_METRIC_LEGEND_BBOX_TO_ANCHOR: tuple[float, float] = (0.5, 0.955)
TRUNCATION_METRIC_LEGEND_NCOL: int = 3

STATIC_LEVEL_LEGEND_LOC: str = "lower center"
STATIC_LEVEL_LEGEND_BBOX_TO_ANCHOR: tuple[float, float] = (0.5, 1.02)
STATIC_LEVEL_LEGEND_NCOL: int = 2
STATIC_LEVEL_LEGEND_FONT_SCALE: float = 0.95

PULSE_SCHEDULE_COLOR: str = "C4"
PULSE_SCHEDULE_ALPHA: float = 0.75

TRUNCATION_METRIC_STYLES: dict[str, dict[str, object]] = {
    "energy_rmse": {"color": "C0", "marker": "s"},
    "j_abs_error": {"color": "C1", "marker": "^"},
    "zeta_abs_error": {"color": "C2", "marker": "d"},
}

MODEL_ALPHAS: dict[str, float] = {
    "circuit": MODEL_ALPHA_CIRCUIT,
    "duffing": MODEL_ALPHA_DUFFING,
    "effective": MODEL_ALPHA_EFFECTIVE,
}
MODEL_COLORS: dict[str, str] = {
    "circuit": "C0",
    "duffing": "C1",
    "effective": "C2",
}
MODEL_LINESTYLES: dict[str, str] = {
    "circuit": "-",
    "duffing": "-",
    "effective": "-",
}

ENERGY_LEVEL_ALPHAS: tuple[float, ...] = (1.0, 0.72, 0.48, 0.32, 0.22, 0.16)
FALLBACK_LEVEL_ALPHA: float = 0.12

SINGLE_COLUMN_FIGURE_HEIGHTS: dict[str, float] = {
    "cz": 2.3,
    "runtime": 2.2,
    "static_main": 3.1,
    "static_raw_energies": 2.7,
    "static_overlaps": 2.35,
    "static_amplitudes": 9.4,
    "leakage_flow": 2.7,
}
SINGLE_COLUMN_STACK_LAYOUTS: dict[str, tuple[float, float]] = {
    "rx_populations": (1.0, 1.0),
    "rx_diagnostics": (1.35, 1.25),
    "truncation_single_model": (1.35, 1.15),
    "truncation_combined": (1.15, 1.1),
}

BENCHMARK_STYLE_STACKS: dict[str, tuple[str, ...]] = {
    "paper": ("benchmark-base", "benchmark-paper"),
    "presentation": ("benchmark-base", "benchmark-presentation"),
}
