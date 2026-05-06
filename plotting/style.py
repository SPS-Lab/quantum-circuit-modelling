"""Shared Matplotlib styling for benchmark plots."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import matplotlib as mpl
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

DEFAULT_PLOT_FONT_SIZE: float = 22.0 # Should be > 18 and < 25 w current style
MODEL_ALPHA_CIRCUIT: float = 1.0
MODEL_ALPHA_DUFFING: float = 0.98
MODEL_ALPHA_EFFECTIVE: float = 0.98
# Controls vertical separation between the top model legend and subplots.
MODEL_LEGEND_BBOX_TO_ANCHOR: tuple[float, float] = (0.5, 1.01)
BENCHMARK_TIGHT_LAYOUT_RECT: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.93)
# Controls spacing between subplot panels for all benchmark figures.
BENCHMARK_TIGHT_LAYOUT_H_PAD: float = 1.2
BENCHMARK_TIGHT_LAYOUT_W_PAD: float = 0.9
# Truncation level-legend controls.
TRUNCATION_LEVEL_LEGEND_MAX_ITEMS: int = 5
TRUNCATION_LEVEL_LEGEND_NCOL: int = 2
TRUNCATION_LEVEL_LEGEND_BBOX_TO_ANCHOR: tuple[float, float] = (0.5, 1.02)
TRUNCATION_LEVEL_LEGEND_FONT_SCALE: float = 0.68
TRUNCATION_LEVEL_LEGEND_TITLE_FONT_SCALE: float = 0.78
TRUNCATION_LEVEL_LEGEND_SHOW_ON_DIFF: bool = False
# Static-spectrum level legend (E1/E2/E3/lower levels) controls.
STATIC_LEVEL_LEGEND_LOC: str = "lower center"
STATIC_LEVEL_LEGEND_BBOX_TO_ANCHOR: tuple[float, float] = (0.5, 1.02)
STATIC_LEVEL_LEGEND_NCOL: int = 2
STATIC_LEVEL_LEGEND_FONT_SCALE: float = 0.72
PULSE_SCHEDULE_COLOR: str = "C4"
PULSE_SCHEDULE_LINEWIDTH: float = 1.8
PULSE_SCHEDULE_ALPHA: float = 0.75

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


def blend_colors(color_a: str | tuple[float, float, float], color_b: str | tuple[float, float, float], weight_b: float) -> tuple[float, float, float]:
    """Blend two colors in RGB with `weight_b` assigned to `color_b`."""
    wb = float(np.clip(weight_b, 0.0, 1.0))
    wa = 1.0 - wb
    a = np.asarray(mcolors.to_rgb(color_a), dtype=float)
    b = np.asarray(mcolors.to_rgb(color_b), dtype=float)
    return tuple(np.clip(wa * a + wb * b, 0.0, 1.0))


def lighten_color(color: str | tuple[float, float, float], amount: float) -> tuple[float, float, float]:
    """Blend a color toward white by `amount`."""
    return blend_colors(color, (1.0, 1.0, 1.0), amount)


def model_color(model: str) -> str:
    """Shared color for a model trace."""
    return MODEL_COLORS[model]


def model_level_color(base_color: str | tuple[float, float, float], model: str) -> tuple[float, float, float]:
    """Color that preserves a level hue while nudging it toward the model color."""
    if model == "circuit":
        return mcolors.to_rgb(base_color)
    return blend_colors(base_color, model_color(model), 0.55)


def model_plot_kwargs(model: str) -> dict[str, object]:
    """Shared line style for a model trace."""
    return {
        "alpha": MODEL_ALPHAS[model],
        "linestyle": MODEL_LINESTYLES[model],
        "color": MODEL_COLORS[model],
    }


def model_legend_handles() -> list[Line2D]:
    """Legend handles that encode model identity consistently across plots."""
    return [
        Line2D([0], [0], linewidth=2.2, label="circuit", **model_plot_kwargs("circuit")),
        Line2D([0], [0], linewidth=2.2, label="duffing", **model_plot_kwargs("duffing")),
        Line2D([0], [0], linewidth=2.2, label="effective", **model_plot_kwargs("effective")),
    ]


def pulse_schedule_plot_kwargs(*, alpha: float | None = None) -> dict[str, object]:
    """Shared style for plotted pulse schedules/flux tracks."""
    return {
        "color": PULSE_SCHEDULE_COLOR,
        "linewidth": PULSE_SCHEDULE_LINEWIDTH,
        "alpha": PULSE_SCHEDULE_ALPHA if alpha is None else float(alpha),
    }


@contextmanager
def benchmark_plot_style(font_size: float = DEFAULT_PLOT_FONT_SIZE) -> Iterator[None]:
    """Temporarily apply shared font sizing across benchmark plots."""
    size = float(font_size)
    with mpl.rc_context(
        rc={
            "font.size": size,
            "axes.titlesize": size,
            "axes.labelsize": size,
            "xtick.labelsize": size * 0.9,
            "ytick.labelsize": size * 0.9,
            "legend.fontsize": size * 0.9,
            "figure.titlesize": size * 1.1,
        }
    ):
        yield
