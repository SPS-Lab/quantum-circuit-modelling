"""Shared plotting theme for benchmark figures."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ACTIVE_BENCHMARK_STYLE: str = "paper"

ACM_SIGCONF_COLUMN_WIDTH_PT: float = 241.14749
ACM_SIGCONF_TEXT_WIDTH_PT: float = 506.295
_TEX_POINTS_PER_INCH: float = 72.27

BENCHMARK_TIGHT_LAYOUT_RECT: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.92)
# In unit scale of font size, default 1.08
BENCHMARK_TIGHT_LAYOUT_PAD: float = 0.5
BENCHMARK_TIGHT_LAYOUT_H_PAD: float = 0.0
BENCHMARK_TIGHT_LAYOUT_W_PAD: float = 0.5

FIGURE_LEGEND_BBOX_TO_ANCHOR: tuple[float, float] = (0.5, 0.985)

COLUMN_TITLE_PAD: float = 11
COLUMN_TITLE_FONTSIZE_EXTRA: float = 2

STATIC_LEVEL_LEGEND_LOC: str = "upper center"
STATIC_LEVEL_LEGEND_BBOX_TO_ANCHOR: tuple[float, float] = (0.5, 1.3)
STATIC_LEVEL_LEGEND_NCOL: int = 3

TRUNCATION_METRIC_LEGEND_NCOL: int = 3

PULSE_SCHEDULE_COLOR: str = "C4"
PULSE_SCHEDULE_ALPHA: float = 0.75
PULSE_BACKGROUND_ALPHA: float = 0.3
PULSE_BACKGROUND_FILL_ALPHA: float = 0.1

REFERENCE_LINE_COLOR: str = "0.35"
REFERENCE_LINEWIDTH: float = 1.0
PRIMARY_LEVEL_LINEWIDTH: float = 1.7
SECONDARY_LEVEL_LINEWIDTH: float = 0.9
ANCILLARY_LEVEL_LINEWIDTH: float = 1.1
COMPARISON_LINEWIDTH: float = 1.2

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

# (width_scale, height_inches)
_FIGURE_SPECS: dict[str, tuple[float, float]] = {
    "cz": (0.85, 1.5),
    "runtime": (1.0, 1.2),
    "static_main": (1.0, 2.9),
    "static_raw_energies": (1.0, 2.35),
    "static_overlaps": (1.0, 2.1),
    "static_amplitudes": (1.0, 9.4),
    "leakage_flow": (1.0, 3.5),
}

# row_count -> (width_scale, height_inches)
_STACKED_FIGURE_SPECS: dict[str, dict[int, tuple[float, float]]] = {
    "rx_populations": {
        2: (1.0, 2.0),
    },
    "rx_diagnostics": {
        3: (1.0, 3.0),
    },
    "truncation_single_model": {
        1: (1.0, 1.25),
        2: (1.0, 2.1),
        3: (1.0, 3.0),
    },
    "truncation_combined": {
        3: (1.0, 3.2),
    },
}


def benchmark_style_paths() -> list[str]:
    """Return the active repo-owned mplstyle files."""
    return [str(_STYLE_DIR / f"{name}.mplstyle") for name in _BENCHMARK_STYLE_STACKS[ACTIVE_BENCHMARK_STYLE]]


def single_column_width_inches() -> float:
    """Return the ACM sigconf single-column width in inches."""
    return ACM_SIGCONF_COLUMN_WIDTH_PT / _TEX_POINTS_PER_INCH


def figure_size(name: str) -> tuple[float, float]:
    """Return a named paper figure size in inches."""
    width_scale, height_inches = _FIGURE_SPECS[name]
    return (width_scale * single_column_width_inches(), height_inches)


def stacked_figure_size(name: str, row_count: int) -> tuple[float, float]:
    """Return an explicit named stacked paper figure size in inches."""
    if row_count <= 0:
        raise ValueError(f"row_count must be positive, got {row_count}")
    try:
        width_scale, height_inches = _STACKED_FIGURE_SPECS[name][row_count]
    except KeyError as exc:
        raise ValueError(f"No stacked figure size recipe for {name!r} with row_count={row_count}") from exc
    return (width_scale * single_column_width_inches(), height_inches)


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


def add_model_figure_legend(
    fig: plt.Figure,
    *,
    handles: list[Line2D] | None = None,
    ncol: int = 3,
    bbox_to_anchor: tuple[float, float] = FIGURE_LEGEND_BBOX_TO_ANCHOR,
) -> None:
    """Add the shared model legend to a figure."""
    fig.legend(
        handles=model_legend_handles() if handles is None else handles,
        loc="upper center",
        ncol=ncol,
        bbox_to_anchor=bbox_to_anchor,
    )


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

def add_column_title(axes: plt.Axes, title: str) -> None:
    """Add larger title to top Axes of column"""
    axes.set_title(title, pad=COLUMN_TITLE_PAD, fontsize=plt.rcParams['axes.titlesize']+COLUMN_TITLE_FONTSIZE_EXTRA)

def benchmark_tight_layout(
    fig: plt.Figure,
) -> None:
    """Apply the shared tight_layout policy for benchmark figures."""
    fig.tight_layout(
        rect=BENCHMARK_TIGHT_LAYOUT_RECT,
        pad=BENCHMARK_TIGHT_LAYOUT_PAD,
        h_pad=BENCHMARK_TIGHT_LAYOUT_H_PAD,
        w_pad=BENCHMARK_TIGHT_LAYOUT_W_PAD
    )


def pulse_schedule_plot_kwargs(*, alpha: float | None = None) -> dict[str, object]:
    """Shared style for plotted pulse schedules and flux tracks."""
    return {
        "color": PULSE_SCHEDULE_COLOR,
        "alpha": PULSE_SCHEDULE_ALPHA if alpha is None else float(alpha),
    }


def save_benchmark_figure(
    fig: plt.Figure,
    outfile: Path,
    *,
    bbox_inches: str | None = None,
    pad_inches: float | None = None,
) -> None:
    """Persist a benchmark figure and close it."""
    outfile.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, object] = {"format": "pdf"}
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
    if pad_inches is not None:
        save_kwargs["pad_inches"] = pad_inches
    fig.savefig(outfile, **save_kwargs)
    plt.close(fig)


@contextmanager
def benchmark_plot_style() -> Iterator[None]:
    """Apply the active repo-owned benchmark style stack with warnings as errors."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with plt.style.context(benchmark_style_paths()):
            yield
