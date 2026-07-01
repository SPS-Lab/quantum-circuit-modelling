"""
Animate a mechanical harmonic oscillator next to its parallel-LC analogue.

This is a visual animation driven by analytic sinusoidal motion, not a
numerical time-domain simulation.

Using the node-flux coordinate phi for a parallel LC oscillator gives

    C * phi_ddot + (1 / L) * phi = 0

which matches the mechanical oscillator equation

    m * x_ddot + k * x = 0.

With this choice of variables, the mechanical displacement x is in phase with
the inductor flux phi and with the inductor current i_L = phi / L. The
capacitor charge q_C = C * phi_dot is quarter-cycle shifted, which is why the
capacitor plate charge is strongest as the mechanical mass passes equilibrium.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle


FPS = 30
DURATION = 8.0
FRAMES = int(FPS * DURATION)


def spring_points(x0: float, x1: float, y: float, turns: int = 10, amplitude: float = 0.18) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(x0, x1, 4 * turns + 3)
    ys = np.full_like(xs, y)
    if xs.size > 2:
        offsets = np.zeros(xs.size - 2)
        offsets[0::2] = amplitude
        offsets[1::2] = -amplitude
        ys[1:-1] += offsets
    return xs, ys


def charge_color(level: float) -> tuple[str, str]:
    if level >= 0.0:
        return "#cf3f3f", "#2b6cb0"
    return "#2b6cb0", "#cf3f3f"


def export_frame_numbers(total_frames: int, step: int) -> list[int]:
    if total_frames <= 0:
        return []
    if step <= 0:
        raise ValueError("step must be positive")

    frame_numbers = list(range(0, total_frames, step))
    if frame_numbers[-1] != total_frames - 1:
        frame_numbers.append(total_frames - 1)
    return frame_numbers


def save_pdf_frames(
    fig: plt.Figure,
    artists: dict[str, object],
    output_dir: Path,
    frame_numbers: list[int],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for export_index, frame_number in enumerate(frame_numbers):
        update(frame_number, artists)
        fig.canvas.draw()
        outpath = output_dir / f"frame_{export_index:03d}.pdf"
        fig.savefig(outpath, format="pdf", bbox_inches="tight")
        saved_paths.append(outpath)

    return saved_paths


def build_figure() -> tuple[plt.Figure, dict[str, object]]:
    fig, (ax_circuit, ax_mech) = plt.subplots(1, 2, figsize=(10.6, 5.2), gridspec_kw={"wspace": 0.1})
    fig.patch.set_facecolor("#ffffff")
    fig.subplots_adjust(top=0.88, bottom=0.035, left=0.055, right=0.97)

    for ax in (ax_circuit, ax_mech):
        ax.set_facecolor("#ffffff")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    ax_circuit.set_xlim(0, 10)
    ax_circuit.set_ylim(1.3, 10)
    ax_circuit.set_aspect("equal")
    ax_circuit.set_title("Parallel LC Oscillator", fontsize=16, pad=0)

    bus_color = "#3c3c3c"
    current_color = "#C0392B"
    flux_color = "#2F9E44"
    charge_top = "#cf3f3f"
    charge_bottom = "#2b6cb0"

    x_left = 3.0
    x_right = 7.0
    y_bottom = 2.0
    y_top = 8.0
    coil_y0 = 3.0
    coil_y1 = 7.0
    coil_turn_centers = np.linspace(3.55, 6.45, 6)
    plate_y_top = 5.45
    plate_y_bottom = 4.55
    plate_half_width = 0.82

    ax_circuit.add_line(Line2D([x_left, x_right], [y_top, y_top], lw=3.0, color=bus_color))
    ax_circuit.add_line(Line2D([x_left, x_right], [y_bottom, y_bottom], lw=3.0, color=bus_color))
    ax_circuit.add_line(Line2D([x_left, x_left], [y_top, coil_turn_centers[-1] + 0.24], lw=3.0, color=bus_color))
    ax_circuit.add_line(Line2D([x_left, x_left], [coil_turn_centers[0] - 0.24, y_bottom], lw=3.0, color=bus_color))
    ax_circuit.add_line(Line2D([x_right, x_right], [y_top, plate_y_top], lw=3.0, color=bus_color))
    ax_circuit.add_line(Line2D([x_right, x_right], [plate_y_bottom, y_bottom], lw=3.0, color=bus_color))

    for center_y in coil_turn_centers:
        ax_circuit.add_patch(
            Ellipse(
                (x_left, center_y),
                width=1.12,
                height=0.42,
                fill=False,
                lw=2.6,
                edgecolor=bus_color,
            )
        )

    top_plate = Line2D([x_right - plate_half_width, x_right + plate_half_width], [plate_y_top, plate_y_top], lw=4.0, color=charge_top)
    bottom_plate = Line2D([x_right - plate_half_width, x_right + plate_half_width], [plate_y_bottom, plate_y_bottom], lw=4.0, color=charge_bottom)
    ax_circuit.add_line(top_plate)
    ax_circuit.add_line(bottom_plate)

    top_plate_fill = Rectangle((x_right - 0.92, plate_y_top - 0.24), 1.84, 0.48, facecolor=charge_top, alpha=0.15, edgecolor="none")
    bottom_plate_fill = Rectangle((x_right - 0.92, plate_y_bottom - 0.24), 1.84, 0.48, facecolor=charge_bottom, alpha=0.15, edgecolor="none")
    ax_circuit.add_patch(top_plate_fill)
    ax_circuit.add_patch(bottom_plate_fill)

    current_arrow_top = FancyArrowPatch((2.15, 6.1), (2.15, 6.9), arrowstyle="-|>", mutation_scale=18, lw=2.4, color=current_color)
    current_arrow_bottom = FancyArrowPatch((2.15, 3.1), (2.15, 3.9), arrowstyle="-|>", mutation_scale=18, lw=2.4, color=current_color)
    ax_circuit.add_patch(current_arrow_top)
    ax_circuit.add_patch(current_arrow_bottom)

    flux_core = FancyBboxPatch(
        (x_left - 0.19, coil_y0 + 0.35),
        0.38,
        coil_y1 - coil_y0 - 0.7,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        facecolor=flux_color,
        edgecolor="none",
        alpha=0.08,
    )
    ax_circuit.add_patch(flux_core)

    flux_arrows = []
    flux_arrow_specs = [(-0.24, 0.14), (0.0, 0.0), (0.24, -0.14)]
    for x_offset, curvature in flux_arrow_specs:
        arrow = FancyArrowPatch(
            (x_left + x_offset, 3.8),
            (x_left + x_offset, 6.2),
            connectionstyle=f"arc3,rad={curvature}",
            arrowstyle="-|>",
            mutation_scale=16,
            lw=2.0,
            color=flux_color,
            alpha=0.12,
        )
        flux_arrows.append(arrow)
        ax_circuit.add_patch(arrow)

    flux_loops = []
    flux_loop_specs = [
        {
            "forward": ((x_left - 0.34, 6.15), (x_left - 0.28, 3.85), "arc3,rad=1.22"),
            "reverse": ((x_left - 0.28, 3.85), (x_left - 0.34, 6.15), "arc3,rad=-1.22"),
        },
        {
            "forward": ((x_left - 0.12, 4.1), (x_left - 0.18, 5.9), "arc3,rad=-0.92"),
            "reverse": ((x_left - 0.18, 5.9), (x_left - 0.12, 4.1), "arc3,rad=0.92"),
        },
        {
            "forward": ((x_left + 0.34, 6.15), (x_left + 0.28, 3.85), "arc3,rad=-1.22"),
            "reverse": ((x_left + 0.28, 3.85), (x_left + 0.34, 6.15), "arc3,rad=1.22"),
        },
        {
            "forward": ((x_left + 0.18, 5.9), (x_left + 0.12, 4.1), "arc3,rad=-0.92"),
            "reverse": ((x_left + 0.12, 4.1), (x_left + 0.18, 5.9), "arc3,rad=0.92"),
        },
    ]
    for spec in flux_loop_specs:
        start, end, connection_style = spec["forward"]
        loop = FancyArrowPatch(
            start,
            end,
            connectionstyle=connection_style,
            arrowstyle="-|>",
            mutation_scale=13,
            lw=1.8,
            color=flux_color,
            alpha=0.18,
        )
        flux_loops.append(loop)
        ax_circuit.add_patch(loop)

    top_charge_text = ax_circuit.text(x_right + 1.2, plate_y_top, "+", ha="center", va="center", fontsize=24, color=charge_top, alpha=0.2)
    bottom_charge_text = ax_circuit.text(x_right + 1.2, plate_y_bottom, "-", ha="center", va="center", fontsize=24, color=charge_bottom, alpha=0.2)

    ax_circuit.text(1.4, 6.3, r"$I$", fontsize=15, color=current_color)
    ax_circuit.text(0.5, 5.1, r"$\Phi$", fontsize=15, color=flux_color)
    ax_circuit.text(1.45, 5.1, r"$L$", fontsize=15, color="#6b4f4f", weight="bold")
    ax_circuit.text(8.4, 4.9, r"$C$", fontsize=15, color="#6b4f4f", weight="bold")

    ax_mech.set_xlim(0, 10)
    ax_mech.set_ylim(1.3, 10)
    ax_mech.set_aspect("equal")
    ax_mech.set_title("Mechanical Oscillator", fontsize=16, pad=0)

    y_track = 4.9
    wall_x = 1.2
    mass_width = 1.45
    mass_height = 1.5
    eq_left = 5.0
    amplitude = 1.55

    ax_mech.add_patch(Rectangle((0.45, 3.35), 0.45, 3.1, facecolor="#8d8d8d", edgecolor="none"))
    ax_mech.add_line(Line2D([0.9, 8.8], [y_track - 0.95, y_track - 0.95], lw=2.0, color="#c7c0b4"))
    ax_mech.add_line(Line2D([eq_left + mass_width / 2.0, eq_left + mass_width / 2.0], [2.1, 7.6], lw=1.4, ls="--", color="#d0c7b8"))
    ax_mech.text(eq_left + mass_width / 2.0, 1.6, "equilibrium", ha="center", va="center", fontsize=10, color="#857d73")

    spring_line = Line2D([], [], lw=3.0, color="#7a4e2d")
    ax_mech.add_line(spring_line)
    mass = Rectangle((eq_left, y_track - mass_height / 2.0), mass_width, mass_height, facecolor="#d6a84f", edgecolor="#6c5228", lw=2.0)
    ax_mech.add_patch(mass)

    displacement_arrow = FancyArrowPatch((eq_left + mass_width / 2.0, 7.9), (eq_left + mass_width / 2.0, 7.9), arrowstyle="<|-|>", mutation_scale=16, lw=2.0, color="#8b5e34")
    force_arrow = FancyArrowPatch((eq_left + mass_width / 2.0, 6.95), (eq_left + mass_width / 2.0, 6.95), arrowstyle="-|>", mutation_scale=18, lw=2.2, color="#b5522d")
    ax_mech.add_patch(displacement_arrow)
    ax_mech.add_patch(force_arrow)
    force_label = ax_mech.text(eq_left + mass_width / 2.0, 7.28, r"$F$", fontsize=15, color="#b5522d", ha="center", va="bottom")
    ax_mech.text(5.5, 8.3, r"$x$", fontsize=15, color="#8b5e34")
    ax_mech.text(1.88, 5.3, r"$k$", fontsize=15, color="#8b5e34")

    artists = {
        "ax_circuit": ax_circuit,
        "ax_mech": ax_mech,
        "top_plate": top_plate,
        "bottom_plate": bottom_plate,
        "top_plate_fill": top_plate_fill,
        "bottom_plate_fill": bottom_plate_fill,
        "current_arrow_top": current_arrow_top,
        "current_arrow_bottom": current_arrow_bottom,
        "flux_core": flux_core,
        "flux_arrows": flux_arrows,
        "flux_arrow_specs": flux_arrow_specs,
        "flux_loops": flux_loops,
        "flux_loop_specs": flux_loop_specs,
        "top_charge_text": top_charge_text,
        "bottom_charge_text": bottom_charge_text,
        "spring_line": spring_line,
        "mass": mass,
        "displacement_arrow": displacement_arrow,
        "force_arrow": force_arrow,
        "force_label": force_label,
        "wall_x": wall_x,
        "y_track": y_track,
        "mass_width": mass_width,
        "eq_left": eq_left,
        "amplitude": amplitude,
    }
    return fig, artists


def update(frame: int, artists: dict[str, object]) -> list[object]:
    phase = 2.0 * np.pi * frame / FRAMES
    position = float(np.cos(phase))
    inductor_current = position
    capacitor_charge = float(-np.sin(phase))

    eq_left = float(artists["eq_left"])
    amplitude = float(artists["amplitude"])
    mass_width = float(artists["mass_width"])
    y_track = float(artists["y_track"])
    wall_x = float(artists["wall_x"])

    mass_left = eq_left + amplitude * position
    mass = artists["mass"]
    assert isinstance(mass, Rectangle)
    mass.set_x(mass_left)

    spring_line = artists["spring_line"]
    assert isinstance(spring_line, Line2D)
    xs, ys = spring_points(wall_x + 0.45, mass_left, y_track)
    spring_line.set_data(xs, ys)

    displacement_arrow = artists["displacement_arrow"]
    assert isinstance(displacement_arrow, FancyArrowPatch)
    center_x = mass_left + mass_width / 2.0
    displacement_arrow.set_positions((eq_left + mass_width / 2.0, 7.9), (center_x, 7.9))

    force_arrow = artists["force_arrow"]
    force_label = artists["force_label"]
    assert isinstance(force_arrow, FancyArrowPatch)
    force_length = 0.2 + 1.1 * abs(position)
    force_direction = -1.0 if position >= 0.0 else 1.0
    force_y = 6.95
    force_start = center_x - 0.5 * force_length * force_direction
    force_end = center_x + 0.5 * force_length * force_direction
    force_arrow.set_positions((force_start, force_y), (force_end, force_y))
    force_arrow.set_alpha(0.18 + 0.78 * abs(position))
    force_arrow.set_linewidth(1.5 + 1.1 * abs(position))
    force_label.set_position((center_x, 7.28))
    force_label.set_alpha(0.35 + 0.65 * abs(position))

    arrow_length = 0.55 + 1.25 * abs(inductor_current)
    current_direction = -1.0 if inductor_current >= 0.0 else 1.0
    field_direction = current_direction
    top_center = 6.5
    bottom_center = 3.5

    current_arrow_top = artists["current_arrow_top"]
    current_arrow_bottom = artists["current_arrow_bottom"]
    assert isinstance(current_arrow_top, FancyArrowPatch)
    assert isinstance(current_arrow_bottom, FancyArrowPatch)

    current_arrow_top.set_positions((2.15, top_center - 0.5 * arrow_length * current_direction), (2.15, top_center + 0.5 * arrow_length * current_direction))
    current_arrow_bottom.set_positions((2.15, bottom_center - 0.5 * arrow_length * current_direction), (2.15, bottom_center + 0.5 * arrow_length * current_direction))

    accent_alpha = 0.2 + 0.8 * abs(inductor_current)
    current_arrow_top.set_alpha(accent_alpha)
    current_arrow_bottom.set_alpha(accent_alpha)

    flux_core = artists["flux_core"]
    flux_arrows = artists["flux_arrows"]
    flux_arrow_specs = artists["flux_arrow_specs"]
    assert isinstance(flux_core, FancyBboxPatch)
    assert isinstance(flux_arrows, list)
    assert isinstance(flux_arrow_specs, list)
    flux_core.set_alpha(0.06 + 0.22 * abs(inductor_current))
    for arrow, (x_offset, curvature) in zip(flux_arrows, flux_arrow_specs):
        assert isinstance(arrow, FancyArrowPatch)
        x_position = 3.0 + x_offset
        field_length = 0.95 + 1.15 * abs(inductor_current)
        horizontal_bend = 0.10 if curvature != 0.0 else 0.0
        start = (x_position - horizontal_bend, 5.0 - 0.5 * field_length)
        end = (x_position + horizontal_bend, 5.0 + 0.5 * field_length)
        if field_direction > 0.0:
            arrow.set_positions(start, end)
        else:
            arrow.set_positions(end, start)
        arrow.set_alpha(0.12 + 0.7 * abs(inductor_current))
        arrow.set_linewidth(1.4 + 1.2 * abs(inductor_current))

    flux_loops = artists["flux_loops"]
    flux_loop_specs = artists["flux_loop_specs"]
    assert isinstance(flux_loops, list)
    assert isinstance(flux_loop_specs, list)
    for loop, spec in zip(flux_loops, flux_loop_specs):
        assert isinstance(loop, FancyArrowPatch)
        if field_direction > 0.0:
            start, end, connection_style = spec["forward"]
        else:
            start, end, connection_style = spec["reverse"]
        loop.set_connectionstyle(connection_style)
        loop.set_positions(start, end)
        loop.set_alpha(0.08 + 0.52 * abs(inductor_current))
        loop.set_linewidth(1.2 + 1.1 * abs(inductor_current))

    top_plate = artists["top_plate"]
    bottom_plate = artists["bottom_plate"]
    top_plate_fill = artists["top_plate_fill"]
    bottom_plate_fill = artists["bottom_plate_fill"]
    top_charge_text = artists["top_charge_text"]
    bottom_charge_text = artists["bottom_charge_text"]
    assert isinstance(top_plate, Line2D)
    assert isinstance(bottom_plate, Line2D)
    assert isinstance(top_plate_fill, Rectangle)
    assert isinstance(bottom_plate_fill, Rectangle)

    top_color, bottom_color = charge_color(capacitor_charge)
    charge_alpha = 0.12 + 0.72 * abs(capacitor_charge)
    top_plate.set_color(top_color)
    bottom_plate.set_color(bottom_color)
    top_plate_fill.set_facecolor(top_color)
    bottom_plate_fill.set_facecolor(bottom_color)
    top_plate_fill.set_alpha(charge_alpha)
    bottom_plate_fill.set_alpha(charge_alpha)

    charge_fontsize = 20 + 10 * abs(capacitor_charge)
    top_charge_text.set_text("+" if capacitor_charge >= 0.0 else "-")
    bottom_charge_text.set_text("-" if capacitor_charge >= 0.0 else "+")
    top_charge_text.set_fontsize(charge_fontsize)
    bottom_charge_text.set_fontsize(charge_fontsize)
    top_charge_text.set_alpha(charge_alpha)
    bottom_charge_text.set_alpha(charge_alpha)
    top_charge_text.set_color(top_color)
    bottom_charge_text.set_color(bottom_color)

    return [
        mass,
        spring_line,
        displacement_arrow,
        force_arrow,
        force_label,
        current_arrow_top,
        current_arrow_bottom,
        top_plate,
        bottom_plate,
        top_plate_fill,
        bottom_plate_fill,
        top_charge_text,
        bottom_charge_text,
        flux_core,
        *flux_arrows,
        *flux_loops,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=Path, help="Optional path to save the animation, e.g. demo.gif or demo.mp4.")
    parser.add_argument(
        "--save-pdf-frames",
        type=Path,
        help="Optional directory to save numbered PDF frames for stepping through in a presentation.",
    )
    parser.add_argument(
        "--pdf-step",
        type=int,
        default=10,
        help="Export every Nth animation frame to PDF. The final frame is always included.",
    )
    parser.add_argument("--fps", type=int, default=FPS, help="Frames per second for playback and saving.")
    parser.add_argument("--seconds", type=float, default=DURATION, help="Animation duration in seconds.")
    parser.add_argument("--no-show", action="store_true", help="Build the animation without opening an interactive window.")
    return parser.parse_args()


def main() -> None:
    global FRAMES

    args = parse_args()
    FRAMES = max(2, int(args.fps * args.seconds))

    fig, artists = build_figure()
    animation = FuncAnimation(
        fig,
        update,
        frames=FRAMES,
        interval=1000 / args.fps,
        blit=True,
        fargs=(artists,),
    )

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        if args.save.suffix.lower() == ".gif":
            animation.save(args.save, writer=PillowWriter(fps=args.fps))
        else:
            animation.save(args.save, fps=args.fps)

    if args.save_pdf_frames:
        frame_numbers = export_frame_numbers(FRAMES, args.pdf_step)
        save_pdf_frames(fig, artists, args.save_pdf_frames, frame_numbers)

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
