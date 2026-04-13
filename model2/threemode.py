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
    heff_spin_to_lab_hamiltonian,
    plot_compare_model1_model2_vs_flux,
)
from model2.core import (
    computational_state_indices,
    computational_subspace_block,
    coupler_frequency,
    three_mode_hamiltonian,
    three_mode_hamiltonian_from_kwargs,
    three_mode_hamiltonian_stack_vs_flux,
)
from model2.dynamics import propagate_piecewise
from model2.effective import (
    build_dressed_effective_computational_stack,
    build_dressed_effective_stack,
    extract_model1_parameters_from_4x4_stack,
    lowdin_orthonormalize_columns,
)
from model2.plots import (
    plot_three_mode_cz_phase_accumulation,
    plot_three_mode_energy_levels,
    plot_three_mode_energy_levels_vs_flux,
    plot_three_mode_zz_exchange_vs_flux,
)
from model2.hamiltonian_types import ThreeModeHamiltonianCommonKwargs, ThreeModeHamiltonianKwargs

__all__ = [
    "ThreeModeHamiltonianCommonKwargs",
    "ThreeModeHamiltonianKwargs",
    "three_mode_hamiltonian",
    "three_mode_hamiltonian_from_kwargs",
    "three_mode_hamiltonian_stack_vs_flux",
    "coupler_frequency",
    "computational_state_indices",
    "computational_subspace_block",
    "lowdin_orthonormalize_columns",
    "build_dressed_effective_stack",
    "build_dressed_effective_computational_stack",
    "extract_model1_parameters_from_4x4_stack",
    "dressed_computational_energies",
    "exchange_and_zz_from_4x4_eigenvalues",
    "heff_spin_to_lab_hamiltonian",
    "plot_three_mode_cz_phase_accumulation",
    "plot_three_mode_zz_exchange_vs_flux",
    "plot_three_mode_energy_levels",
    "plot_three_mode_energy_levels_vs_flux",
    "plot_compare_model1_model2_vs_flux",
    "propagate_piecewise",
]


_ThreeModeHamiltonianKwargs = ThreeModeHamiltonianCommonKwargs


if __name__ == "__main__":
    _dir = Path(__file__).resolve().parent
    _common: ThreeModeHamiltonianCommonKwargs = {
        "w_1": 5.0,
        "w_2": 5.0,
        "alpha_1": -0.5,
        "alpha_c": -0.0,
        "alpha_2": -0.5,
        "g_1c": 0.50,
        "g_2c": 0.50,
        "nlevels_qubit": 2,
        "nlevels_coupler": 2,
    }
#    plot_three_mode_energy_levels(
#        outfile=str(_dir / "three_mode_energy_levels.pdf"),
#        w_c=5.2,
#        **_common,
#    )
    from_flux = 0.0
    to_flux = 1.0
    flux = np.linspace(from_flux, to_flux, 41)
#    plot_three_mode_energy_levels_vs_flux(
#        flux,
#        wc0=5.2,
#        A=0.25,
#        outfile=str(_dir / "three_mode_energy_levels_vs_flux.pdf"),
#        n_show=16,
#        **_common,
#    )
#    plot_three_mode_zz_exchange_vs_flux(
#        flux,
#        wc0=5.2,
#        A=0.25,
#        outfile=str(_dir / "three_mode_ZZ_exchange_vs_flux.pdf"),
#        **_common,
#    )
#    flux = np.linspace(0.0, 1.0, 80)
#    plot_compare_model1_model2_vs_flux(
#        flux,
#        outfile=str(_dir / f"model1_vs_eff_model2_energy_vs_flux_{from_flux}to{to_flux}.pdf"),
#        subtract_ground=True,
#        verbose=True,
#        wc0=5.0,
#        A=1.0,
#        plot_eff2_levels=True,
#        n_model2_levels=8,
#        dressed_selection_mode="continuous",
#        **_common,
#    )
    
    plot_compare_model1_model2_vs_flux(
        flux,
        wc0=5.0,
        A=1.0,
        model1_params={
            "w1": lambda phi: 5.0 + 0.2*np.cos(2*np.pi*phi),
            "w2": lambda phi: 5.1 + 0.15*np.cos(2*np.pi*phi + 0.1),
            "J":  lambda phi: 0.01 + 0.003*np.cos(2*np.pi*phi),
            "zeta": 0.002,
        },
        **_common,
    )
    
    plot_three_mode_cz_phase_accumulation(
        flux=0.4667, wc0=6.0, A=1.0,
        w_1=5.0, w_2=5.2,
        alpha_1=-0.3, alpha_c=-0.25, alpha_2=-0.32,
        g_1c=0.08, g_2c=0.075,
        nlevels_qubit=3, nlevels_coupler=3,
        outfile="three_mode_cz_phase_accumulation.pdf",
    )


    # Test 3-dim H
    #nlevels_qubit = 2
    #nlevels_coupler = 2
    #N = nlevels_qubit ** 2 * nlevels_coupler
    #n_fluxes = 3
    #H = np.arange(n_fluxes * 2**N * 2**N).reshape(n_fluxes, 2**N, 2**N)
    #print(f"{H.shape=}")
#
    #block = computational_subspace_block(H, nlevels_qubit, nlevels_coupler)
    
    
    
