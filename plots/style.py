"""Shared Matplotlib styling for benchmark plots."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import matplotlib as mpl

DEFAULT_PLOT_FONT_SIZE: float = 13.0


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
