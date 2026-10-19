"""Observable extraction from simulated amplitude histories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CzDynamicsObservables:
    """CZ observables derived from computational and optional monitored amplitudes."""

    computational_amplitudes: np.ndarray
    populations_11: np.ndarray
    leakage_11: np.ndarray
    conditional_phase: np.ndarray
    monitor_population_11: np.ndarray


def extract_cz_observables(
    comp_amp: np.ndarray,
    *,
    monitor_amplitudes: np.ndarray | None = None,
) -> CzDynamicsObservables:
    """Analyze amplitudes in computational order |00>, |01>, |10>, |11>.

    Both arrays use axes (time, output basis row, initial computational state).
    Optional monitored rows contribute their total population from initial |11>.
    """
    amp = np.asarray(comp_amp, dtype=complex)
    if amp.ndim != 3 or amp.shape[1:] != (4, 4):
        raise ValueError(f"comp_amp must be (n_time, 4, 4), got {amp.shape}")

    populations_11 = np.abs(amp[:, :, 3]) ** 2
    leakage_11 = 1.0 - np.sum(populations_11, axis=1)
    leakage_11 = np.clip(np.real(leakage_11), 0.0, 1.0)

    d00 = amp[:, 0, 0]
    d01 = amp[:, 1, 1]
    d10 = amp[:, 2, 2]
    d11 = amp[:, 3, 3]
    den = d01 * d10
    eps = 1e-15
    den_safe = np.where(np.abs(den) < eps, eps + 0.0j, den)
    cond_phase = np.unwrap(np.angle((d00 * d11) / den_safe))

    monitor_population = np.zeros(amp.shape[0], dtype=float)
    if monitor_amplitudes is not None:
        monitor = np.asarray(monitor_amplitudes, dtype=complex)
        if (
            monitor.ndim != 3
            or monitor.shape[0] != amp.shape[0]
            or monitor.shape[2] != 4
        ):
            raise ValueError(
                "monitor_amplitudes must have shape (n_time, n_monitored, 4)"
            )
        monitor_population = np.sum(np.abs(monitor[:, :, 3]) ** 2, axis=1)

    return CzDynamicsObservables(
        computational_amplitudes=amp,
        populations_11=np.asarray(populations_11, dtype=float),
        leakage_11=np.asarray(leakage_11, dtype=float),
        conditional_phase=np.asarray(cond_phase, dtype=float),
        monitor_population_11=monitor_population,
    )
