"""Shared Matplotlib styling for benchmark plots."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import matplotlib as mpl
from matplotlib.lines import Line2D

DEFAULT_PLOT_FONT_SIZE: float = 22.0 # Should be > 18 and < 25 w current style
MODEL_ALPHA_CIRCUIT: float = 1.0
MODEL_ALPHA_DUFFING: float = 0.72
MODEL_ALPHA_EFFECTIVE: float = 0.45
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

MODEL_ALPHAS: dict[str, float] = {
    "circuit": MODEL_ALPHA_CIRCUIT,
    "duffing": MODEL_ALPHA_DUFFING,
    "effective": MODEL_ALPHA_EFFECTIVE,
}
MODEL_LINESTYLES: dict[str, str] = {
    "circuit": "-",
    "duffing": "--",
    "effective": "-.",
}


def model_plot_kwargs(model: str) -> dict[str, object]:
    """Shared line style for a model trace."""
    return {
        "alpha": MODEL_ALPHAS[model],
        "linestyle": MODEL_LINESTYLES[model],
    }


def model_legend_handles() -> list[Line2D]:
    """Legend handles that encode model identity consistently across plots."""
    return [
        Line2D([0], [0], color="k", linewidth=2.2, label="circuit", **model_plot_kwargs("circuit")),
        Line2D([0], [0], color="k", linewidth=2.2, label="duffing", **model_plot_kwargs("duffing")),
        Line2D([0], [0], color="k", linewidth=2.2, label="effective", **model_plot_kwargs("effective")),
    ]


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
