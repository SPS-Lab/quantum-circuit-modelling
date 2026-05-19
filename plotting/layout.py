"""Project-specific figure geometry recipes for benchmark plots."""

from __future__ import annotations

from plotting.constants import (
    ACM_SIGCONF_COLUMN_WIDTH_PT,
    ACM_SIGCONF_TEXT_WIDTH_PT,
    SINGLE_COLUMN_FIGURE_HEIGHTS,
    SINGLE_COLUMN_STACK_LAYOUTS,
    TEX_POINTS_PER_INCH,
)


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


def named_single_column_figure_size(layout_name: str) -> tuple[float, float]:
    """Build a named single-column figure size in inches."""
    return single_column_figure_size(SINGLE_COLUMN_FIGURE_HEIGHTS[layout_name])


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


def named_stacked_single_column_figure_size(layout_name: str, row_count: int) -> tuple[float, float]:
    """Build a named stacked single-column figure size from a row count."""
    row_height_inches, extra_height_inches = SINGLE_COLUMN_STACK_LAYOUTS[layout_name]
    return stacked_figure_size(
        row_count,
        column_span=1,
        row_height_inches=row_height_inches,
        extra_height_inches=extra_height_inches,
    )
