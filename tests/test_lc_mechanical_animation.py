from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib.pyplot as plt


def load_animation_module():
    module_path = Path(__file__).resolve().parents[1] / "harmonic" / "lc_mechanical_animation.py"
    spec = spec_from_file_location("lc_mechanical_animation", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_animation_builds_and_updates_multiple_frames() -> None:
    animation_module = load_animation_module()
    fig, artists = animation_module.build_figure()

    first_frame_artists = animation_module.update(0, artists)
    quarter_frame_artists = animation_module.update(animation_module.FRAMES // 4, artists)

    assert len(first_frame_artists) >= 10
    assert len(quarter_frame_artists) == len(first_frame_artists)
    assert "x ~ i_L" in artists["relation_text"].get_text()
    assert artists["top_charge_text"].get_text() in {"+", "-"}
    assert artists["bottom_charge_text"].get_text() in {"+", "-"}

    plt.close(fig)
