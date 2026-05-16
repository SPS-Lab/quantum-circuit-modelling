"""Shared sweep-control helpers for fixed-bus static benchmark model builders."""

from __future__ import annotations

import numpy as np

from study_config import SystemParams


def resolve_static_sweep_values(
    flux_values: np.ndarray,
    *,
    system_params: SystemParams,
    sweep_target: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve per-flux q0 flux, q1 flux, and coupler frequency arrays.

    `flux_values` is interpreted according to `sweep_target`:
    - `q0`: sweep q0 flux, keep q1 flux and the fixed bus frequency constant.
    - `q1`: sweep q1 flux, keep q0 flux and the fixed bus frequency constant.
    """
    flux_arr = np.asarray(flux_values, dtype=float).ravel()
    target = str(sweep_target).strip().lower()
    wc = np.full_like(flux_arr, float(system_params.c.E_osc), dtype=float)

    if target == "q0":
        q0_flux = np.asarray(flux_arr, dtype=float).ravel()
        q1_flux = np.full_like(flux_arr, float(system_params.q1.flux), dtype=float)
    elif target == "q1":
        q0_flux = np.full_like(flux_arr, float(system_params.q0.flux), dtype=float)
        q1_flux = np.asarray(flux_arr, dtype=float).ravel()
    else:
        raise ValueError(f"Unsupported sweep_target {sweep_target!r}")

    return q0_flux, q1_flux, wc
