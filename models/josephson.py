"""Josephson-energy and CPB-related helpers."""

from __future__ import annotations

import numpy as np



def flux_dependent_EJ(
    *,
    EJ_max: float,
    flux_bias: np.ndarray | float,
    d: float
) -> np.ndarray:
    """Return SQUID effective Josephson energy vs reduced flux bias.

    Formula:
    ``EJ(phi) = EJ_max * sqrt(cos^2(pi*phi) + d^2 sin^2(pi*phi))``.
    """
    x = np.pi * np.asarray(flux_bias, dtype=float)
    return float(EJ_max) * np.sqrt(np.cos(x) ** 2 + (float(d) ** 2) * np.sin(x) ** 2)
