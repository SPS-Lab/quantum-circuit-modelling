"""Project-specific figure geometry recipes for benchmark plots."""

from __future__ import annotations

ACM_SIGCONF_COLUMN_WIDTH_PT: float = 241.14749
ACM_SIGCONF_TEXT_WIDTH_PT: float = 506.295
TEX_POINTS_PER_INCH: float = 72.27

SINGLE_COLUMN_FIGURE_HEIGHTS: dict[str, float] = {
    "cz": 2.3,
    "runtime": 2.2,
    "static_main": 3.1,
    "static_raw_energies": 2.7,
    "static_overlaps": 2.35,
    "static_amplitudes": 9.4,
    "leakage_flow": 6.2,
}
SINGLE_COLUMN_STACK_LAYOUTS: dict[str, tuple[float, float]] = {
    "rx_populations": (1.0, 1.0),
    "rx_diagnostics": (1.35, 1.25),
    "truncation_single_model": (1.35, 1.15),
    "truncation_combined": (1.15, 1.1),
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
