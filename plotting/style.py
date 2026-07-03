"""Shared plotting theme for benchmark figures."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.lines import Line2D

ACTIVE_BENCHMARK_STYLE: str = "paper"

ACM_SIGCONF_COLUMN_WIDTH_PT: float = 241.14749
ACM_SIGCONF_TEXT_WIDTH_PT: float = 506.295
_TEX_POINTS_PER_INCH: float = 72.27
_CM_PER_INCH: float = 2.54

BEAMER_169_SLIDE_WIDTH_CM: float = 16.0
BEAMER_169_SLIDE_HEIGHT_CM: float = 9.0

BENCHMARK_TIGHT_LAYOUT_RECT: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
# In unit scale of font size, default 1.08
BENCHMARK_TIGHT_LAYOUT_PAD: float = 0.6
BENCHMARK_TIGHT_LAYOUT_H_PAD: float = 0.4
BENCHMARK_TIGHT_LAYOUT_W_PAD: float = 1.08
BENCHMARK_TIGHT_LAYOUT_TOP_GAP: float = 0.01

FIGURE_LEGEND_TOP_MARGIN_INCHES: float = 0.01

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
_CURRENT_BENCHMARK_STYLE: ContextVar[str] = ContextVar(
    "current_benchmark_style",
    default=ACTIVE_BENCHMARK_STYLE,
)

PRESENTATION_FIGURE_WIDTH_SCALE_MAX: float = 0.8
PRESENTATION_FIGURE_HEIGHT_INCHES_MAX: float = 2.5

# style -> figure_name -> (width_scale, height_inches)
_FIGURE_SPECS: dict[str, dict[str, tuple[float, float]]] = {
    "paper": {
        "cz": (0.95, 1.7),
        "runtime": (1.0, 1.65),
        "static_main": (1.0, 2.9),
        "static_raw_energies": (1.0, 2.35),
        "static_overlaps": (1.25, 2.1),
        "static_amplitudes": (1.0, 9.4),
        "leakage_flow": (1.0, 3.8),
    },
    "presentation": {
        "cz": (PRESENTATION_FIGURE_WIDTH_SCALE_MAX, PRESENTATION_FIGURE_HEIGHT_INCHES_MAX),
        "runtime": (PRESENTATION_FIGURE_WIDTH_SCALE_MAX, PRESENTATION_FIGURE_HEIGHT_INCHES_MAX),
        "static_main": (PRESENTATION_FIGURE_WIDTH_SCALE_MAX, PRESENTATION_FIGURE_HEIGHT_INCHES_MAX),
        "static_raw_energies": (PRESENTATION_FIGURE_WIDTH_SCALE_MAX, PRESENTATION_FIGURE_HEIGHT_INCHES_MAX),
        "static_overlaps": (PRESENTATION_FIGURE_WIDTH_SCALE_MAX, PRESENTATION_FIGURE_HEIGHT_INCHES_MAX),
        "static_amplitudes": (PRESENTATION_FIGURE_WIDTH_SCALE_MAX, PRESENTATION_FIGURE_HEIGHT_INCHES_MAX),
        "leakage_flow": (PRESENTATION_FIGURE_WIDTH_SCALE_MAX, PRESENTATION_FIGURE_HEIGHT_INCHES_MAX),
    },
}

# style -> row_count -> (width_scale, height_inches)
_STACKED_FIGURE_SPECS: dict[str, dict[str, dict[int, tuple[float, float]]]] = {
    "paper": {
        "rx_populations": {
            3: (1.0, 3.4),
        },
        "rx_diagnostics": {
            2: (1.0, 2.5),
        },
        "truncation_single_model": {
            1: (1.0, 2.2),
            2: (1.0, 2.7),
            3: (1.0, 4.0),
        },
        "truncation_combined": {
            3: (1.0, 4.2),
        },
    },
    "presentation": {
        "rx_populations": {
            3: (PRESENTATION_FIGURE_WIDTH_SCALE_MAX, PRESENTATION_FIGURE_HEIGHT_INCHES_MAX),
        },
        "rx_diagnostics": {
            2: (PRESENTATION_FIGURE_WIDTH_SCALE_MAX, PRESENTATION_FIGURE_HEIGHT_INCHES_MAX),
        },
        "truncation_single_model": {
            1: (PRESENTATION_FIGURE_WIDTH_SCALE_MAX, PRESENTATION_FIGURE_HEIGHT_INCHES_MAX),
            2: (PRESENTATION_FIGURE_WIDTH_SCALE_MAX, PRESENTATION_FIGURE_HEIGHT_INCHES_MAX),
            3: (PRESENTATION_FIGURE_WIDTH_SCALE_MAX, PRESENTATION_FIGURE_HEIGHT_INCHES_MAX),
        },
        "truncation_combined": {
            3: (PRESENTATION_FIGURE_WIDTH_SCALE_MAX, PRESENTATION_FIGURE_HEIGHT_INCHES_MAX),
        },
    },
}


def benchmark_style_names() -> tuple[str, ...]:
    """Return the repo-owned benchmark styles to materialize for every plot."""
    return tuple(_BENCHMARK_STYLE_STACKS)


def current_benchmark_style() -> str:
    """Return the style active while the current figure is being built."""
    return _CURRENT_BENCHMARK_STYLE.get()


def benchmark_style_paths(style_name: str | None = None) -> list[str]:
    """Return the repo-owned mplstyle files for a benchmark style."""
    style_key = ACTIVE_BENCHMARK_STYLE if style_name is None else str(style_name)
    return [str(_STYLE_DIR / f"{name}.mplstyle") for name in _BENCHMARK_STYLE_STACKS[style_key]]


def benchmark_style_outfile(outfile: Path, style_name: str) -> Path:
    """Return the materialized outfile path for a specific benchmark style."""
    return outfile.with_name(f"{outfile.stem}_{style_name}{outfile.suffix}")


def single_column_width_inches() -> float:
    """Return the ACM sigconf single-column width in inches."""
    return ACM_SIGCONF_COLUMN_WIDTH_PT / _TEX_POINTS_PER_INCH


def beamer_slide_width_inches() -> float:
    """Return the Beamer 16:9 slide width in inches."""
    return BEAMER_169_SLIDE_WIDTH_CM / _CM_PER_INCH


def beamer_slide_height_inches() -> float:
    """Return the Beamer 16:9 slide height in inches."""
    return BEAMER_169_SLIDE_HEIGHT_CM / _CM_PER_INCH


def benchmark_reference_width_inches(style_name: str | None = None) -> float:
    """Return the reference layout width for a benchmark style."""
    style_key = _CURRENT_BENCHMARK_STYLE.get() if style_name is None else str(style_name)
    if style_key == "paper":
        return single_column_width_inches()
    if style_key == "presentation":
        return beamer_slide_width_inches()
    raise KeyError(f"Unknown benchmark style {style_key!r}")


def figure_size(name: str) -> tuple[float, float]:
    """Return a named benchmark figure size in inches."""
    style_key = _CURRENT_BENCHMARK_STYLE.get()
    width_scale, height_inches = _FIGURE_SPECS[style_key][name]
    return (
        width_scale * benchmark_reference_width_inches(),
        height_inches,
    )


def stacked_figure_size(name: str, row_count: int) -> tuple[float, float]:
    """Return an explicit named stacked benchmark figure size in inches."""
    if row_count <= 0:
        raise ValueError(f"row_count must be positive, got {row_count}")
    try:
        width_scale, height_inches = _STACKED_FIGURE_SPECS[_CURRENT_BENCHMARK_STYLE.get()][name][row_count]
    except KeyError as exc:
        raise ValueError(f"No stacked figure size recipe for {name!r} with row_count={row_count}") from exc
    return (
        width_scale * benchmark_reference_width_inches(),
        height_inches,
    )


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
        Line2D([0], [0], label="Circuit (reference)", **model_plot_kwargs("circuit")),
        Line2D([0], [0], label="Duffing", **model_plot_kwargs("duffing")),
        Line2D([0], [0], label="Effective", **model_plot_kwargs("effective")),
    ]


def figure_legend_bbox_to_anchor(fig: plt.Figure) -> tuple[float, float]:
    """Return a figure-legend anchor with a fixed physical top margin."""
    _, fig_height_inches = fig.get_size_inches()
    if fig_height_inches <= 0.0:
        return (0.5, 0.985)
    return (0.5, 1.0 - FIGURE_LEGEND_TOP_MARGIN_INCHES / float(fig_height_inches))


def add_model_figure_legend(
    fig: plt.Figure,
    *,
    handles: list[Line2D] | None = None,
    ncol: int = 3,
    bbox_to_anchor: tuple[float, float] | None = None,
) -> Artist:
    """Add the shared model legend to a figure."""
    return fig.legend(
        handles=model_legend_handles() if handles is None else handles,
        loc="upper center",
        ncol=ncol,
        bbox_to_anchor=figure_legend_bbox_to_anchor(fig) if bbox_to_anchor is None else bbox_to_anchor,
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
    *,
    reserve_artists: list[Artist] | None = None,
    w_pad: float | None = None,
) -> None:
    """Apply the shared tight_layout policy for benchmark figures."""
    rect = list(BENCHMARK_TIGHT_LAYOUT_RECT)
    if reserve_artists:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        top_limit = rect[3]
        for artist in reserve_artists:
            bbox = artist.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
            top_limit = min(top_limit, float(bbox.y0) - BENCHMARK_TIGHT_LAYOUT_TOP_GAP)
        rect[3] = max(rect[1] + 0.1, top_limit)
    fig.tight_layout(
        rect=tuple(rect),
        pad=BENCHMARK_TIGHT_LAYOUT_PAD,
        h_pad=BENCHMARK_TIGHT_LAYOUT_H_PAD,
        w_pad=BENCHMARK_TIGHT_LAYOUT_W_PAD if w_pad is None else float(w_pad)
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
    style_name: str,
) -> None:
    """Persist a benchmark figure and close it."""
    outfile = benchmark_style_outfile(outfile, style_name)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, object] = {"format": "pdf"}
    fig.savefig(outfile, **save_kwargs)
    plt.close(fig)


def render_benchmark_figures(
    outfile: Path,
    build_figure: Callable[[], plt.Figure],
) -> None:
    """Render and persist the same benchmark figure in each repo-owned style."""
    for style_name in benchmark_style_names():
        with benchmark_plot_style(style_name):
            fig = build_figure()
            save_benchmark_figure(fig, outfile, style_name=style_name)


@contextmanager
def benchmark_plot_style(style_name: str | None = None) -> Iterator[None]:
    """Apply a repo-owned benchmark style stack with warnings as errors."""
    style_key = ACTIVE_BENCHMARK_STYLE if style_name is None else str(style_name)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        token = _CURRENT_BENCHMARK_STYLE.set(style_key)
        try:
            with plt.style.context(benchmark_style_paths(style_key)):
                yield
        finally:
            _CURRENT_BENCHMARK_STYLE.reset(token)
