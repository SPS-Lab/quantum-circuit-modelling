"""Shared Matplotlib styling and figure geometry for benchmark plots."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

ACTIVE_BENCHMARK_STYLE: str = "paper"
ACM_SIGCONF_COLUMN_WIDTH_PT: float = 241.14749
ACM_SIGCONF_TEXT_WIDTH_PT: float = 506.295
TEX_POINTS_PER_INCH: float = 72.27
MODEL_ALPHA_CIRCUIT: float = 1.0
MODEL_ALPHA_DUFFING: float = 0.98
MODEL_ALPHA_EFFECTIVE: float = 0.98
# Controls vertical separation between the top model legend and subplots.
MODEL_LEGEND_BBOX_TO_ANCHOR: tuple[float, float] = (0.5, 1.01)
BENCHMARK_TIGHT_LAYOUT_RECT: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.93)
# Controls spacing between subplot panels for all benchmark figures.
BENCHMARK_TIGHT_LAYOUT_H_PAD: float = 1.2
BENCHMARK_TIGHT_LAYOUT_W_PAD: float = 0.9
TRUNCATION_METRIC_LEGEND_BBOX_TO_ANCHOR: tuple[float, float] = (0.5, 0.955)
TRUNCATION_METRIC_LEGEND_NCOL: int = 3
# Static-spectrum level legend (E1/E2/E3/lower levels) controls.
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

_STYLE_DIR = Path(__file__).with_name("styles")
_STYLE_STACKS: dict[str, tuple[str, ...]] = {
    "paper": ("benchmark-base", "benchmark-paper"),
    "presentation": ("benchmark-base", "benchmark-presentation"),
}


def tex_pt_to_inches(points: float) -> float:
    """Convert TeX points to inches."""
    return float(points) / TEX_POINTS_PER_INCH


def single_column_width_inches() -> float:
    """Return the ACM sigconf single-column width in inches."""
    return tex_pt_to_inches(ACM_SIGCONF_COLUMN_WIDTH_PT)


def text_width_inches() -> float:
    """Return the ACM sigconf full text width in inches."""
    return tex_pt_to_inches(ACM_SIGCONF_TEXT_WIDTH_PT)


def single_column_figure_size(height_inches: float) -> tuple[float, float]:
    """Build a single-column figure size in inches."""
    return (single_column_width_inches(), float(height_inches))


def text_width_figure_size(height_inches: float) -> tuple[float, float]:
    """Build a full-text-width figure size in inches."""
    return (text_width_inches(), float(height_inches))


def stacked_figure_size(
    row_count: int,
    *,
    column_span: int = 1,
    row_height_inches: float,
    extra_height_inches: float = 0.0,
) -> tuple[float, float]:
    """Build a stacked-panel figure size from a semantic row count."""
    if row_count <= 0:
        raise ValueError(f"row_count must be positive, got {row_count}")
    if column_span == 1:
        width_inches = single_column_width_inches()
    elif column_span == 2:
        width_inches = text_width_inches()
    else:
        raise ValueError(f"column_span must be 1 or 2, got {column_span}")
    height_inches = float(extra_height_inches) + float(row_count) * float(row_height_inches)
    return (width_inches, height_inches)


def benchmark_style_paths() -> list[str]:
    """Return the active repo-owned mplstyle files."""
    style_names = _STYLE_STACKS[ACTIVE_BENCHMARK_STYLE]
    return [str(_STYLE_DIR / f"{style_name}.mplstyle") for style_name in style_names]


def active_font_size() -> float:
    """Return the active Matplotlib base font size."""
    return float(mpl.rcParams["font.size"])


def scaled_font_size(scale: float, *, minimum: float | None = None) -> float:
    """Scale the active Matplotlib base font size with an optional floor."""
    size = active_font_size() * float(scale)
    if minimum is not None:
        size = max(size, float(minimum))
    return float(size)


def model_linewidth() -> float:
    """Return the active line width for model traces."""
    return float(mpl.rcParams["lines.linewidth"])


def truncation_metric_linewidth() -> float:
    """Return the active line width for truncation metric traces."""
    return 0.9 * model_linewidth()


def pulse_schedule_linewidth() -> float:
    """Return the active line width for pulse schedules and flux tracks."""
    return 0.95 * model_linewidth()


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


def energy_level_alpha(level_index: int) -> float:
    """Shared alpha for an energy level trace."""
    idx = int(level_index)
    if idx < 0:
        raise ValueError(f"level_index must be non-negative, got {level_index}")
    if idx < len(ENERGY_LEVEL_ALPHAS):
        return float(ENERGY_LEVEL_ALPHAS[idx])
    return FALLBACK_LEVEL_ALPHA


def model_level_color(base_color: str | tuple[float, float, float], model: str) -> tuple[float, float, float]:
    """Preserve a level hue without re-encoding model identity in color."""
    del model
    return mcolors.to_rgb(base_color)


def model_plot_kwargs(
    model: str,
    *,
    color: str | tuple[float, float, float] | None = None,
) -> dict[str, object]:
    """Shared line style for a model trace."""
    resolved_color = MODEL_COLORS[model] if color is None else color
    return {
        "alpha": MODEL_ALPHAS[model],
        "linestyle": MODEL_LINESTYLES[model],
        "color": resolved_color,
        "linewidth": model_linewidth(),
    }


def model_legend_handles() -> list[Line2D]:
    """Legend handles that encode model identity consistently across plots."""
    return [
        Line2D([0], [0], label="circuit", **model_plot_kwargs("circuit")),
        Line2D([0], [0], label="duffing", **model_plot_kwargs("duffing")),
        Line2D([0], [0], label="effective", **model_plot_kwargs("effective")),
    ]


def truncation_metric_legend_handles() -> list[Line2D]:
    """Legend handles for truncation benchmark metric traces."""
    return [
        Line2D([0], [0], label=r"$RMSE_{E,\mathrm{comp}}$", **truncation_metric_plot_kwargs("energy_rmse")),
        Line2D([0], [0], label=r"$|\Delta J|$", **truncation_metric_plot_kwargs("j_abs_error")),
        Line2D([0], [0], label=r"$|\Delta \zeta|$", **truncation_metric_plot_kwargs("zeta_abs_error")),
    ]


def pulse_schedule_plot_kwargs(*, alpha: float | None = None) -> dict[str, object]:
    """Shared style for plotted pulse schedules/flux tracks."""
    return {
        "color": PULSE_SCHEDULE_COLOR,
        "linewidth": pulse_schedule_linewidth(),
        "alpha": PULSE_SCHEDULE_ALPHA if alpha is None else float(alpha),
    }


def truncation_metric_plot_kwargs(metric: str) -> dict[str, object]:
    """Shared style for truncation benchmark metric traces."""
    return {
        **TRUNCATION_METRIC_STYLES[metric],
        "linewidth": truncation_metric_linewidth(),
    }


def apply_benchmark_grid(ax, *, visible: bool = True) -> None:
    """Apply the shared benchmark grid treatment to an axes."""
    ax.grid(visible)


@contextmanager
def benchmark_plot_style() -> Iterator[None]:
    """Apply the active repo-owned benchmark style stack."""
    with plt.style.context(benchmark_style_paths()):
        yield
