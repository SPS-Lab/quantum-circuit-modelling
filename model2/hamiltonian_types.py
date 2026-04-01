"""Shared type definitions for model2."""

from __future__ import annotations

from typing import TypedDict


class ThreeModeHamiltonianCommonKwargs(TypedDict):
    """Common fixed keyword arguments for three-mode Hamiltonians,
    excluding flux dependent ``w_c``."""

    w_1: float
    w_2: float
    alpha_1: float
    alpha_c: float
    alpha_2: float
    g_1c: float
    g_2c: float
    nlevels_qubit: int
    nlevels_coupler: int


class ThreeModeHamiltonianKwargs(ThreeModeHamiltonianCommonKwargs):
    """Keyword arguments required by ``three_mode_hamiltonian``."""

    w_c: float
