"""Shared sweep-control helpers for static benchmark model builders."""

from __future__ import annotations

import numpy as np

from models.three_mode import coupler_frequency
from study_config import CouplerFrequencyConfig, SystemParams


def resolve_static_sweep_values(
    flux_values: np.ndarray,
    *,
    system_params: SystemParams,
    coupler_frequency_config: CouplerFrequencyConfig,
    sweep_target: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve per-flux q1 flux, q2 flux, and coupler frequency arrays.

    `flux_values` is interpreted according to `sweep_target`:
    - `coupler`: sweep coupler frequency via `wc(phi)`, keep q1/q2 flux fixed.
    - `q1`: sweep q1 flux, keep q2 flux and coupler frequency fixed.
    - `q2`: sweep q2 flux, keep q1 flux and coupler frequency fixed.
    """
    flux_arr = np.asarray(flux_values, dtype=float).ravel()
    target = str(sweep_target).strip().lower()

    if target == "coupler":
        q1_flux = np.full_like(flux_arr, float(system_params.q1.flux), dtype=float)
        q2_flux = np.full_like(flux_arr, float(system_params.q2.flux), dtype=float)
        wc = np.asarray(
            coupler_frequency(
                wc0=float(coupler_frequency_config.wc0),
                A=float(coupler_frequency_config.amplitude),
                flux=flux_arr,
            ),
            dtype=float,
        ).ravel()
    elif target == "q1":
        q1_flux = np.asarray(flux_arr, dtype=float).ravel()
        q2_flux = np.full_like(flux_arr, float(system_params.q2.flux), dtype=float)
        wc = np.full_like(flux_arr, float(coupler_frequency_config.wc0), dtype=float)
    elif target == "q2":
        q1_flux = np.full_like(flux_arr, float(system_params.q1.flux), dtype=float)
        q2_flux = np.asarray(flux_arr, dtype=float).ravel()
        wc = np.full_like(flux_arr, float(coupler_frequency_config.wc0), dtype=float)
    else:
        raise ValueError(f"Unsupported sweep_target {sweep_target!r}")

    return q1_flux, q2_flux, wc
