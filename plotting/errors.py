"""Shared error transformations for benchmark plots."""

from __future__ import annotations

import numpy as np


def sweep_normalized_absolute_error_percent(
    candidate: np.ndarray,
    circuit_reference: np.ndarray,
) -> np.ndarray:
    """Return 100 * |candidate-reference| / max_sweep(|reference|)."""
    candidate_array = np.asarray(candidate, dtype=float)
    reference_array = np.asarray(circuit_reference, dtype=float)
    scale = float(np.max(np.abs(reference_array)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Cannot normalize error by an identically zero or non-finite circuit trace")
    return 100.0 * np.abs(candidate_array - reference_array) / scale
