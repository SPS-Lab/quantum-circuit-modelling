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
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


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


def coil_points(x: float, y0: float, y1: float, loops: int = 5, width: float = 0.28) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 2.0 * np.pi * loops, 250)
    ys = np.linspace(y0, y1, t.size)
    xs = x + width * np.sin(t)
    return xs, ys


def charge_color(level: float) -> tuple[str, str]:
    if level >= 0.0:
        return "#cf3f3f", "#2b6cb0"
    return "#2b6cb0", "#cf3f3f"


def build_figure() -> tuple[plt.Figure, dict[str, object]]:
    fig, (ax_circuit, ax_mech) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"wspace": 0.16})
    fig.patch.set_facecolor("#fbfaf6")

    for ax in (ax_circuit, ax_mech):
        ax.set_facecolor("#fffdf8")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    ax_circuit.set_xlim(0, 10)
    ax_circuit.set_ylim(0, 10)
    ax_circuit.set_aspect("equal")
    ax_circuit.set_title("Parallel LC Oscillator", fontsize=16, pad=14)

    bus_color = "#3c3c3c"
    accent = "#0b6e4f"
    charge_top = "#cf3f3f"
    charge_bottom = "#2b6cb0"

    x_left = 3.0
    x_right = 7.0
    y_bottom = 2.0
    y_top = 8.0
    coil_y0 = 3.0
    coil_y1 = 7.0
    plate_y_top = 5.45
    plate_y_bottom = 4.55
    plate_half_width = 0.82

    ax_circuit.add_line(Line2D([x_left, x_right], [y_top, y_top], lw=3.0, color=bus_color))
    ax_circuit.add_line(Line2D([x_left, x_right], [y_bottom, y_bottom], lw=3.0, color=bus_color))
    ax_circuit.add_line(Line2D([x_left, x_left], [y_top, coil_y1], lw=3.0, color=bus_color))
    ax_circuit.add_line(Line2D([x_left, x_left], [coil_y0, y_bottom], lw=3.0, color=bus_color))
    ax_circuit.add_line(Line2D([x_right, x_right], [y_top, plate_y_top], lw=3.0, color=bus_color))
    ax_circuit.add_line(Line2D([x_right, x_right], [plate_y_bottom, y_bottom], lw=3.0, color=bus_color))

    coil_xs, coil_ys = coil_points(x_left, coil_y0, coil_y1)
    ax_circuit.plot(coil_xs, coil_ys, color=accent, lw=3.2)

    top_plate = Line2D([x_right - plate_half_width, x_right + plate_half_width], [plate_y_top, plate_y_top], lw=4.0, color=charge_top)
    bottom_plate = Line2D([x_right - plate_half_width, x_right + plate_half_width], [plate_y_bottom, plate_y_bottom], lw=4.0, color=charge_bottom)
    ax_circuit.add_line(top_plate)
    ax_circuit.add_line(bottom_plate)

    top_plate_fill = Rectangle((x_right - 0.92, plate_y_top - 0.24), 1.84, 0.48, facecolor=charge_top, alpha=0.15, edgecolor="none")
    bottom_plate_fill = Rectangle((x_right - 0.92, plate_y_bottom - 0.24), 1.84, 0.48, facecolor=charge_bottom, alpha=0.15, edgecolor="none")
    ax_circuit.add_patch(top_plate_fill)
    ax_circuit.add_patch(bottom_plate_fill)

    current_arrow_top = FancyArrowPatch((2.15, 6.1), (2.15, 6.9), arrowstyle="-|>", mutation_scale=18, lw=2.4, color=accent)
    current_arrow_bottom = FancyArrowPatch((2.15, 3.1), (2.15, 3.9), arrowstyle="-|>", mutation_scale=18, lw=2.4, color=accent)
    ax_circuit.add_patch(current_arrow_top)
    ax_circuit.add_patch(current_arrow_bottom)

    flux_rings = []
    for radius in (0.9, 1.25, 1.6):
        ring = Circle((x_left, 5.0), radius=radius, fill=False, lw=2.0, alpha=0.12, color=accent)
        flux_rings.append(ring)
        ax_circuit.add_patch(ring)

    top_charge_text = ax_circuit.text(x_right + 1.2, plate_y_top, "+", ha="center", va="center", fontsize=24, color=charge_top, alpha=0.2)
    bottom_charge_text = ax_circuit.text(x_right + 1.2, plate_y_bottom, "-", ha="center", va="center", fontsize=24, color=charge_bottom, alpha=0.2)

    ax_circuit.text(1.18, 7.95, "inductor current", fontsize=11, color=accent)
    ax_circuit.text(6.1, 6.25, "capacitor charge", fontsize=11, color="#6b4f4f")
    ax_circuit.text(2.45, 5.1, "L", fontsize=13, color=accent, weight="bold")
    ax_circuit.text(8.0, 5.0, "C", fontsize=13, color="#6b4f4f", weight="bold")
    ax_circuit.text(0.55, 0.65, "Choice of analogy: x_mech <-> inductor flux/current", fontsize=11, color="#464646")

    ax_mech.set_xlim(0, 10)
    ax_mech.set_ylim(0, 10)
    ax_mech.set_aspect("equal")
    ax_mech.set_title("Mechanical Oscillator", fontsize=16, pad=14)

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
    ax_mech.add_patch(displacement_arrow)
    ax_mech.text(1.18, 8.3, "position x", fontsize=11, color="#8b5e34")
    ax_mech.text(0.85, 0.65, "Peak position lines up with peak inductor current/flux", fontsize=11, color="#464646")

    relation_text = ax_mech.text(
        0.5,
        -0.12,
        "Analytic phase relation: x(t) and i_L(t) are in phase; capacitor charge q_C(t) is shifted by 90 degrees.",
        transform=ax_mech.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color="#4b4b4b",
    )

    artists = {
        "ax_circuit": ax_circuit,
        "ax_mech": ax_mech,
        "top_plate": top_plate,
        "bottom_plate": bottom_plate,
        "top_plate_fill": top_plate_fill,
        "bottom_plate_fill": bottom_plate_fill,
        "current_arrow_top": current_arrow_top,
        "current_arrow_bottom": current_arrow_bottom,
        "flux_rings": flux_rings,
        "top_charge_text": top_charge_text,
        "bottom_charge_text": bottom_charge_text,
        "spring_line": spring_line,
        "mass": mass,
        "displacement_arrow": displacement_arrow,
        "relation_text": relation_text,
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

    arrow_length = 0.55 + 1.25 * abs(inductor_current)
    direction = 1.0 if inductor_current >= 0.0 else -1.0
    top_center = 6.5
    bottom_center = 3.5

    current_arrow_top = artists["current_arrow_top"]
    current_arrow_bottom = artists["current_arrow_bottom"]
    assert isinstance(current_arrow_top, FancyArrowPatch)
    assert isinstance(current_arrow_bottom, FancyArrowPatch)

    current_arrow_top.set_positions((2.15, top_center - 0.5 * arrow_length * direction), (2.15, top_center + 0.5 * arrow_length * direction))
    current_arrow_bottom.set_positions((2.15, bottom_center - 0.5 * arrow_length * direction), (2.15, bottom_center + 0.5 * arrow_length * direction))

    accent_alpha = 0.2 + 0.8 * abs(inductor_current)
    current_arrow_top.set_alpha(accent_alpha)
    current_arrow_bottom.set_alpha(accent_alpha)

    flux_rings = artists["flux_rings"]
    assert isinstance(flux_rings, list)
    for index, ring in enumerate(flux_rings, start=1):
        assert isinstance(ring, Circle)
        ring.set_alpha((0.1 + 0.16 * index) * abs(inductor_current) + 0.04)
        ring.set_linewidth(1.2 + 1.0 * abs(inductor_current))

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

    relation_text = artists["relation_text"]
    time_fraction = frame / FRAMES
    relation_text.set_text(
        f"phase = {time_fraction:0.2f} cycles   |   x ~ i_L = {inductor_current:+0.2f}   |   q_C = {capacitor_charge:+0.2f}"
    )

    return [
        mass,
        spring_line,
        displacement_arrow,
        current_arrow_top,
        current_arrow_bottom,
        top_plate,
        bottom_plate,
        top_plate_fill,
        bottom_plate_fill,
        top_charge_text,
        bottom_charge_text,
        relation_text,
        *flux_rings,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=Path, help="Optional path to save the animation, e.g. demo.gif or demo.mp4.")
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

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
