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


def test_pdf_frame_export_writes_numbered_files(tmp_path: Path) -> None:
    animation_module = load_animation_module()
    fig, artists = animation_module.build_figure()

    frame_numbers = animation_module.export_frame_numbers(12, 5)
    saved_paths = animation_module.save_pdf_frames(fig, artists, tmp_path, frame_numbers)

    assert frame_numbers == [0, 5, 10, 11]
    assert [path.name for path in saved_paths] == [
        "frame_000.pdf",
        "frame_001.pdf",
        "frame_002.pdf",
        "frame_003.pdf",
    ]
    assert all(path.exists() for path in saved_paths)

    plt.close(fig)
