from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt

from plotting.style import (
    ACM_SIGCONF_COLUMN_WIDTH_PT,
    ACTIVE_BENCHMARK_STYLE,
    BEAMER_169_SLIDE_HEIGHT_CM,
    BEAMER_169_SLIDE_WIDTH_CM,
    beamer_slide_height_inches,
    beamer_slide_width_inches,
    benchmark_reference_width_inches,
    benchmark_plot_style,
    benchmark_style_names,
    benchmark_style_outfile,
    benchmark_style_paths,
    energy_level_alpha,
    figure_size,
    model_legend_handles,
    model_plot_kwargs,
    single_column_width_inches,
    truncation_metric_legend_handles,
    truncation_metric_plot_kwargs,
)


def test_model_plot_kwargs_use_model_color_without_markers() -> None:
    circuit = model_plot_kwargs("circuit")
    duffing = model_plot_kwargs("duffing")
    effective = model_plot_kwargs("effective")

    assert circuit["color"] == "C0"
    assert duffing["color"] == "C1"
    assert effective["color"] == "C2"
    assert "marker" not in circuit
    assert "marker" not in duffing
    assert "marker" not in effective


def test_energy_level_alpha_descends_with_level_index() -> None:
    assert energy_level_alpha(0) == 1.0
    assert energy_level_alpha(1) < energy_level_alpha(0)
    assert energy_level_alpha(2) < energy_level_alpha(1)
    assert energy_level_alpha(12) < energy_level_alpha(2)


def test_model_legend_handles_keep_model_colors() -> None:
    handles = model_legend_handles()

    assert [handle.get_label() for handle in handles] == ["Circuit", "Duffing", "Effective"]
    assert [handle.get_color() for handle in handles] == ["C0", "C1", "C2"]


def test_benchmark_plot_style_uses_shared_default_font_size() -> None:
    base_size = plt.rcParams["font.size"]

    with benchmark_plot_style():
        assert plt.rcParams["font.size"] == 8.5
        assert plt.rcParams["mathtext.fontset"] == "stix"
        assert abs(plt.rcParams["figure.figsize"][0] - single_column_width_inches()) < 1e-6

    assert plt.rcParams["font.size"] == base_size


def test_truncation_metric_handles_match_shared_metric_styles() -> None:
    handles = truncation_metric_legend_handles()
    metrics = ("energy_rmse", "j_abs_error", "zeta_abs_error")

    for handle, metric in zip(handles, metrics):
        kwargs = truncation_metric_plot_kwargs(metric)
        assert handle.get_color() == kwargs["color"]
        assert handle.get_marker() == kwargs["marker"]


def test_benchmark_style_paths_point_to_repo_owned_stylesheets() -> None:
    assert ACTIVE_BENCHMARK_STYLE == "paper"
    assert ACM_SIGCONF_COLUMN_WIDTH_PT == 241.14749
    assert BEAMER_169_SLIDE_WIDTH_CM == 16.0
    assert BEAMER_169_SLIDE_HEIGHT_CM == 9.0
    assert benchmark_style_names() == ("paper", "presentation")
    assert all(Path(path).exists() for path in benchmark_style_paths())
    assert all(Path(path).exists() for style_name in benchmark_style_names() for path in benchmark_style_paths(style_name))


def test_benchmark_style_outfile_adds_style_suffix_before_extension(tmp_path: Path) -> None:
    outfile = tmp_path / "example.pdf"

    assert benchmark_style_outfile(outfile, "paper").name == "example_paper.pdf"
    assert benchmark_style_outfile(outfile, "presentation").name == "example_presentation.pdf"


def test_simple_single_axis_figures_can_be_narrower_than_full_column() -> None:
    assert figure_size("cz")[0] < single_column_width_inches()
    assert figure_size("static_raw_energies")[0] <= single_column_width_inches()
    assert figure_size("runtime")[0] <= single_column_width_inches()


def test_presentation_reference_width_matches_beamer_169_width() -> None:
    assert abs(beamer_slide_width_inches() - (16.0 / 2.54)) < 1e-9
    assert abs(beamer_slide_height_inches() - (9.0 / 2.54)) < 1e-9
    with benchmark_plot_style("presentation"):
        assert abs(benchmark_reference_width_inches() - beamer_slide_width_inches()) < 1e-9
        assert abs(figure_size("runtime")[0] - beamer_slide_width_inches()) < 1e-9
