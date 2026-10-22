"""Reusable flux pulse schedules, with times expressed in ns."""

from __future__ import annotations

import numpy as np


def _time_grid(total_time_ns: float, dt_ns: float) -> np.ndarray:
    total = float(total_time_ns)
    dt = float(dt_ns)
    if total <= 0.0:
        raise ValueError("total_time_ns must be positive")
    if dt <= 0.0:
        raise ValueError("dt_ns must be positive")

    grid = np.arange(0.0, total, dt, dtype=float)
    if grid.size == 0:
        grid = np.array([0.0], dtype=float)
    if abs(grid[-1] - total) > 1e-12:
        grid = np.append(grid, total)
    else:
        grid[-1] = total
    if grid.size < 2:
        grid = np.array([0.0, total], dtype=float)
    diffs = np.diff(grid)
    if np.any(diffs <= 0.0):
        raise ValueError("Internal error: non-increasing time grid")
    return grid


def ramp_hold_ramp_flux_pulse(
    *,
    ramp_time_ns: float,
    hold_time_ns: float,
    dt_ns: float,
    idle_flux: float,
    target_flux: float,
    compress: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a cosine ramp-hold-ramp flux pulse and time grid in ns.

    The returned grid includes both endpoints. Negative hold times are
    clamped to zero, matching the benchmark pulse convention.

    With ``compress=True``, return ``(times, flux_values, counts)``. The time
    grid is unchanged, but consecutive equal interval fluxes are represented
    once, with a positive integer number of steps in ``counts``. Their sum
    is ``len(times) - 1``. The unused terminal flux is omitted. Build one
    Hamiltonian per returned flux and pass ``counts`` to the CPU propagator.
    """
    ramp = float(ramp_time_ns)
    hold = float(max(0.0, hold_time_ns))
    if ramp <= 0.0:
        raise ValueError("ramp_time_ns must be positive")

    total = 2.0 * ramp + hold
    times = _time_grid(total, dt_ns)
    flux = np.empty_like(times, dtype=float)
    delta = float(target_flux) - float(idle_flux)

    t1 = ramp
    t2 = ramp + hold
    for k, t in enumerate(times):
        if t <= t1:
            u = t / ramp
            env = 0.5 * (1.0 - np.cos(np.pi * u))
        elif t <= t2:
            env = 1.0
        else:
            u = (t - t2) / ramp
            env = 0.5 * (1.0 + np.cos(np.pi * u))
        flux[k] = float(idle_flux) + delta * env

    if not compress:
        return times, flux

    interval_flux = flux[:-1]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(interval_flux) != 0) + 1))
    counts = np.diff(np.append(starts, interval_flux.size))
    return times, interval_flux[starts], counts
