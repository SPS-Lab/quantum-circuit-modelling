"""Idle-regime model-1 calibration from model-2 without per-flux parameter labels.

This script:
1) Builds dressed model-2 computational 4x4 Hamiltonians in an idle flux window.
2) Fits quadratic flux models for (w1, w2, J, zeta) directly against model-2 energies.
3) Evaluates train/holdout errors and plots model-1 vs model-2 comparisons.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares

# Repo root so `model1` / `model2` / `toolkit` resolve when executed directly.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model1.heff import heff
from model2.comparison import heff_spin_to_lab_hamiltonian, plot_compare_model1_model2_vs_flux
from model2.core import three_mode_hamiltonian_stack_vs_flux
from model2.effective import build_dressed_effective_computational_stack


def centered_quadratic_values(
    flux_values: np.ndarray,
    coeffs: np.ndarray,
    phi0: float,
) -> np.ndarray:
    """Return values for coeffs=[c2,c1,c0] using (phi-phi0) centered quadratic."""
    c2, c1, c0 = (float(v) for v in coeffs)
    dphi = np.asarray(flux_values, dtype=float) - float(phi0)
    return c2 * dphi * dphi + c1 * dphi + c0


def unpack_theta(
    theta: np.ndarray,
    flux_values: np.ndarray,
    phi0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map flattened coeff vector to parameter curves (w1, w2, J, zeta)."""
    coeffs = np.asarray(theta, dtype=float).reshape(4, 3)
    w1 = centered_quadratic_values(flux_values, coeffs[0], phi0)
    w2 = centered_quadratic_values(flux_values, coeffs[1], phi0)
    j = centered_quadratic_values(flux_values, coeffs[2], phi0)
    zeta = centered_quadratic_values(flux_values, coeffs[3], phi0)
    return w1, w2, j, zeta


def model1_relative_energies_from_theta(
    theta: np.ndarray,
    flux_values: np.ndarray,
    phi0: float,
) -> np.ndarray:
    """Return model-1 eigenenergies relative to ground for each flux."""
    w1, w2, j, zeta = unpack_theta(theta, flux_values, phi0)
    h1_raw = np.asarray(heff(w1, w2, j, zeta), dtype=complex)
    h1 = heff_spin_to_lab_hamiltonian(h1_raw, w1, w2)
    e1 = np.linalg.eigvalsh(h1)
    return e1 - e1[:, :1]


def summarize_error(name: str, err: np.ndarray, mask: np.ndarray) -> str:
    """Human-readable max/RMSE summary on masked rows."""
    e = np.asarray(err, dtype=float)[mask]
    rmse = float(np.sqrt(np.mean(e * e)))
    emax = float(np.max(np.abs(e)))
    return f"{name}: rmse={rmse:.3e}, max_abs={emax:.3e}"


def main() -> None:
    wc0 = 6.0
    A = 1.0
    ham_kwargs = {
        "w_1": 5.0,
        "w_2": 5.2,
        "alpha_1": -0.3,
        "alpha_c": -0.25,
        "alpha_2": -0.32,
        "g_1c": 0.08,
        "g_2c": 0.075,
        "nlevels_qubit": 3,
        "nlevels_coupler": 3,
    }

    phi_idle = 0.20
    half_window = 0.08
    n_flux = 41
    flux = np.linspace(phi_idle - half_window, phi_idle + half_window, n_flux)

    train_mask = (np.arange(n_flux) % 2) == 0
    holdout_mask = ~train_mask

    h2 = three_mode_hamiltonian_stack_vs_flux(
        flux,
        wc0=wc0,
        A=A,
        ham_kwargs=ham_kwargs,
    )
    h2_eff = build_dressed_effective_computational_stack(
        h2,
        nlevels_qubit=int(ham_kwargs["nlevels_qubit"]),
        nlevels_coupler=int(ham_kwargs["nlevels_coupler"]),
    )
    e2_rel = np.linalg.eigvalsh(h2_eff)
    e2_rel = e2_rel - e2_rel[:, :1]

    # Initialize from physics-inspired constants near idle center.
    theta0 = np.array(
        [
            0.0,
            0.0,
            float(ham_kwargs["w_1"]),
            0.0,
            0.0,
            float(ham_kwargs["w_2"]),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    def residuals(theta: np.ndarray) -> np.ndarray:
        e1_rel = model1_relative_energies_from_theta(theta, flux[train_mask], phi_idle)
        # ignore ground level column (always ~0 after subtraction)
        return (e1_rel[:, 1:] - e2_rel[train_mask, 1:]).ravel()

    fit = least_squares(residuals, theta0, method="trf", max_nfev=20000)
    theta_fit = fit.x

    e1_rel = model1_relative_energies_from_theta(theta_fit, flux, phi_idle)
    diff_rel = e1_rel - e2_rel

    print("Direct fit status:")
    print(f"  success={fit.success}, nfev={fit.nfev}, cost={fit.cost:.6e}")
    print("Fitted centered quadratic coefficients [c2, c1, c0]:")
    print(f"  w1   = {theta_fit[0:3]}")
    print(f"  w2   = {theta_fit[3:6]}")
    print(f"  J    = {theta_fit[6:9]}")
    print(f"  zeta = {theta_fit[9:12]}")
    print("\nRelative-energy mismatch vs model-2 dressed 4x4 levels:")
    print(" train ", summarize_error("E rel", diff_rel, train_mask))
    print(" hold  ", summarize_error("E rel", diff_rel, holdout_mask))

    w1_fn = lambda p: centered_quadratic_values(p, theta_fit[0:3], phi_idle)
    w2_fn = lambda p: centered_quadratic_values(p, theta_fit[3:6], phi_idle)
    j_fn = lambda p: centered_quadratic_values(p, theta_fit[6:9], phi_idle)
    zeta_fn = lambda p: centered_quadratic_values(p, theta_fit[9:12], phi_idle)

    outdir = Path(__file__).resolve().parent
    model1_params = {
        "w1": w1_fn,
        "w2": w2_fn,
        "J": j_fn,
        "zeta": zeta_fn,
    }

    evals1_eff, evals2_eff = plot_compare_model1_model2_vs_flux(
        flux,
        outfile=str(outdir / "idle_model1_fit_vs_model2_eff.pdf"),
        subtract_ground=True,
        verbose=False,
        wc0=wc0,
        A=A,
        plot_eff2_levels=True,
        model1_params=model1_params,
        **ham_kwargs,
    )
    print("\nPlot mismatch (sanity check against plotting path):")
    print(" train ", summarize_error("E rel", evals1_eff - evals2_eff, train_mask))
    print(" hold  ", summarize_error("E rel", evals1_eff - evals2_eff, holdout_mask))

    evals1_full, evals2_full = plot_compare_model1_model2_vs_flux(
        flux,
        outfile=str(outdir / "idle_model1_fit_vs_model2_full_low4.pdf"),
        subtract_ground=True,
        verbose=False,
        wc0=wc0,
        A=A,
        plot_eff2_levels=False,
        model1_params=model1_params,
        **ham_kwargs,
    )
    print("\nRelative-energy mismatch vs full model-2 low-4 tracked levels:")
    print(" train ", summarize_error("E rel", evals1_full - evals2_full, train_mask))
    print(" hold  ", summarize_error("E rel", evals1_full - evals2_full, holdout_mask))
    print("\nWrote:")
    print(f"  {outdir / 'idle_model1_fit_vs_model2_eff.pdf'}")
    print(f"  {outdir / 'idle_model1_fit_vs_model2_full_low4.pdf'}")


if __name__ == "__main__":
    main()
