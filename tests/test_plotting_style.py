from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plotting.style import energy_level_alpha, model_legend_handles, model_plot_kwargs


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

    assert [handle.get_label() for handle in handles] == ["circuit", "duffing", "effective"]
    assert [handle.get_color() for handle in handles] == ["C0", "C1", "C2"]
