"""Shared plotting theme for benchmark figures."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ACTIVE_BENCHMARK_STYLE: str = "paper"

ACM_SIGCONF_COLUMN_WIDTH_PT: float = 241.14749
ACM_SIGCONF_TEXT_WIDTH_PT: float = 506.295
_TEX_POINTS_PER_INCH: float = 72.27

BENCHMARK_TIGHT_LAYOUT_RECT: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.93)
BENCHMARK_TIGHT_LAYOUT_H_PAD: float = 1.2
BENCHMARK_TIGHT_LAYOUT_W_PAD: float = 0.9

MODEL_LEGEND_BBOX_TO_ANCHOR: tuple[float, float] = (0.5, 1.01)
TRUNCATION_METRIC_LEGEND_BBOX_TO_ANCHOR: tuple[float, float] = (0.5, 0.955)
TRUNCATION_METRIC_LEGEND_NCOL: int = 3
STATIC_LEVEL_LEGEND_LOC: str = "lower center"
STATIC_LEVEL_LEGEND_BBOX_TO_ANCHOR: tuple[float, float] = (0.5, 1.02)
STATIC_LEVEL_LEGEND_NCOL: int = 2

PULSE_SCHEDULE_COLOR: str = "C4"
PULSE_SCHEDULE_ALPHA: float = 0.75

MODEL_ALPHAS: dict[str, float] = {
    "circuit": 1.0,
    "duffing": 0.98,
    "effective": 0.98,
}
MODEL_COLORS: dict[str, str] = {
    "circuit": "C0",
    "duffing": "C1",
    "effective": "C2",
}
TRUNCATION_METRIC_STYLES: dict[str, dict[str, object]] = {
    "energy_rmse": {"color": "C0", "marker": "s"},
    "j_abs_error": {"color": "C1", "marker": "^"},
    "zeta_abs_error": {"color": "C2", "marker": "d"},
}
ENERGY_LEVEL_ALPHAS: tuple[float, ...] = (1.0, 0.72, 0.48, 0.32, 0.22, 0.16)
FALLBACK_LEVEL_ALPHA: float = 0.12

_STYLE_DIR = Path(__file__).with_name("styles")
_BENCHMARK_STYLE_STACKS: dict[str, tuple[str, ...]] = {
    "paper": ("benchmark-base", "benchmark-paper"),
    "presentation": ("benchmark-base", "benchmark-presentation"),
}
_SINGLE_COLUMN_FIGURE_HEIGHTS: dict[str, float] = {
    "cz": 2.3,
    "runtime": 2.2,
    "static_main": 3.1,
    "static_raw_energies": 2.7,
    "static_overlaps": 2.35,
    "static_amplitudes": 9.4,
    "leakage_flow": 2.7,
}
_STACKED_FIGURE_HEIGHTS: dict[str, tuple[float, float]] = {
    "rx_populations": (1.0, 1.0),
    "rx_diagnostics": (1.35, 1.25),
    "truncation_single_model": (1.35, 1.15),
    "truncation_combined": (1.15, 1.1),
}


def benchmark_style_paths() -> list[str]:
    """Return the active repo-owned mplstyle files."""
    return [str(_STYLE_DIR / f"{name}.mplstyle") for name in _BENCHMARK_STYLE_STACKS[ACTIVE_BENCHMARK_STYLE]]


def single_column_width_inches() -> float:
    """Return the ACM sigconf single-column width in inches."""
    return ACM_SIGCONF_COLUMN_WIDTH_PT / _TEX_POINTS_PER_INCH


def figure_size(name: str) -> tuple[float, float]:
    """Return a named single-column figure size in inches."""
    return (single_column_width_inches(), _SINGLE_COLUMN_FIGURE_HEIGHTS[name])


def stacked_figure_size(name: str, row_count: int) -> tuple[float, float]:
    """Return a named stacked single-column figure size in inches."""
    if row_count <= 0:
        raise ValueError(f"row_count must be positive, got {row_count}")
    row_height_inches, extra_height_inches = _STACKED_FIGURE_HEIGHTS[name]
    return (single_column_width_inches(), extra_height_inches + row_count * row_height_inches)


def energy_level_alpha(level_index: int) -> float:
    """Shared alpha for an energy level trace."""
    idx = int(level_index)
    if idx < 0:
        raise ValueError(f"level_index must be non-negative, got {level_index}")
    if idx < len(ENERGY_LEVEL_ALPHAS):
        return ENERGY_LEVEL_ALPHAS[idx]
    return FALLBACK_LEVEL_ALPHA


def model_color(model: str) -> str:
    """Shared color for a model trace."""
    return MODEL_COLORS[model]


def model_plot_kwargs(
    model: str,
    *,
    color: str | tuple[float, float, float] | None = None,
) -> dict[str, object]:
    """Shared style for a model trace."""
    return {
        "alpha": MODEL_ALPHAS[model],
        "color": MODEL_COLORS[model] if color is None else color,
    }


def model_legend_handles() -> list[Line2D]:
    """Legend handles that encode model identity consistently across plots."""
    return [
        Line2D([0], [0], label="circuit", **model_plot_kwargs("circuit")),
        Line2D([0], [0], label="duffing", **model_plot_kwargs("duffing")),
        Line2D([0], [0], label="effective", **model_plot_kwargs("effective")),
    ]


def truncation_metric_plot_kwargs(metric: str) -> dict[str, object]:
    """Shared style for truncation benchmark metric traces."""
    return dict(TRUNCATION_METRIC_STYLES[metric])


def truncation_metric_legend_handles() -> list[Line2D]:
    """Legend handles for truncation benchmark metric traces."""
    return [
        Line2D([0], [0], label=r"$RMSE_{E,\mathrm{comp}}$", **truncation_metric_plot_kwargs("energy_rmse")),
        Line2D([0], [0], label=r"$|\Delta J|$", **truncation_metric_plot_kwargs("j_abs_error")),
        Line2D([0], [0], label=r"$|\Delta \zeta|$", **truncation_metric_plot_kwargs("zeta_abs_error")),
    ]


def pulse_schedule_plot_kwargs(*, alpha: float | None = None) -> dict[str, object]:
    """Shared style for plotted pulse schedules and flux tracks."""
    return {
        "color": PULSE_SCHEDULE_COLOR,
        "alpha": PULSE_SCHEDULE_ALPHA if alpha is None else float(alpha),
    }


@contextmanager
def benchmark_plot_style() -> Iterator[None]:
    """Apply the active repo-owned benchmark style stack."""
    with plt.style.context(benchmark_style_paths()):
        yield
