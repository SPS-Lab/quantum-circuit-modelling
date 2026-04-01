"""Compatibility facade for the three-mode model utilities.

This module now re-exports focused functionality from:
- ``model2.core``: Hamiltonian/basis helpers
- ``model2.analysis``: ZZ/exchange metrics
- ``model2.plots``: plotting helpers
- ``model2.comparison``: model1-vs-model2 bridge and diagnostics
- ``model2.dynamics``: piecewise propagation
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Repo root (parent of model2/) so `toolkit` and `model2.*` resolve when run from model2/.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model2.analysis import (
    dressed_computational_energies,
    exchange_and_zz_from_4x4_eigenvalues,
)
from model2.comparison import (
    _import_model1_heff,
    heff_spin_to_lab_hamiltonian,
    plot_compare_model1_model2_vs_flux,
)
from model2.core import computational_state_indices, coupler_frequency, three_mode_hamiltonian
from model2.dynamics import propagate_piecewise
from model2.plots import (
    plot_three_mode_energy_levels,
    plot_three_mode_energy_levels_vs_flux,
    plot_three_mode_zz_exchange_vs_flux,
)

__all__ = [
    "three_mode_hamiltonian",
    "coupler_frequency",
    "computational_state_indices",
    "dressed_computational_energies",
    "exchange_and_zz_from_4x4_eigenvalues",
    "heff_spin_to_lab_hamiltonian",
    "plot_three_mode_zz_exchange_vs_flux",
    "plot_three_mode_energy_levels",
    "plot_three_mode_energy_levels_vs_flux",
    "plot_compare_model1_model2_vs_flux",
    "propagate_piecewise",
]


if __name__ == "__main__":
    _dir = Path(__file__).resolve().parent
    _common = dict(
        w_1=5.0,
        w_2=5.0,
        alpha_1=-0.5,
        alpha_c=-0.0,
        alpha_2=-0.5,
        g_1c=0.50,
        g_2c=0.50,
        nlevels_qubit=2,
        nlevels_coupler=2,
    )
#    plot_three_mode_energy_levels(
#        outfile=str(_dir / "three_mode_energy_levels.pdf"),
#        w_c=5.2,
#        **_common,
#    )
    flux = np.linspace(0.0, 1.0, 80)
#    plot_three_mode_energy_levels_vs_flux(
#        flux,
#        wc0=5.2,
#        A=0.25,
#        outfile=str(_dir / "three_mode_energy_levels_vs_flux.pdf"),
#        n_show=16,
#        **_common,
#    )
    plot_three_mode_zz_exchange_vs_flux(
        flux,
        wc0=5.2,
        A=0.25,
        outfile=str(_dir / "three_mode_ZZ_exchange_vs_flux.pdf"),
        **_common,
    )
    flux = np.linspace(0.0, 1.0, 80)
    plot_compare_model1_model2_vs_flux(
        flux,
        outfile=str(_dir / "model1_vs_model2_energy_vs_flux.pdf"),
        subtract_ground=True,
        verbose=True,
        wc0=5.0,
        A=0.0,
        **_common,
    )
