"""Full-range (phi in [0, 1]) cosine surrogate for model-2-derived metrics.

Workflow:
1) Build full model-2 Hamiltonians across flux.
2) Construct dressed computational effective 4x4 Hamiltonians.
3) Extract model-1-style metrics (w1, w2, J, zeta) at each flux.
4) Build cosine surrogates from metric mean/max:
   metric(phi) ~= metric_0 + metric_A * cos(2*pi*phi),
   with metric_0 = mean(metric), metric_A = max(metric) - metric_0.
5) Plot true metrics vs cosine surrogates.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

# Repo root so `model2` / `toolkit` resolve when executed directly.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model2.core import coupler_frequency, three_mode_hamiltonian_stack_vs_flux
from model2.effective import (
    build_dressed_effective_computational_stack,
    extract_model1_parameters_from_4x4_stack,
)


def cosine_from_mean_and_max(
    flux_values: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Build ``v0 + vA cos(2*pi*phi)`` using ``v0=mean(values)``, ``vA=max(values)-v0``."""
    flux_values = np.asarray(flux_values, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    v0 = float(np.mean(values))
    vA = float(np.max(values) - v0)
    model = v0 + vA * np.cos(2 * np.pi * flux_values)
    return model, v0, vA


def cosine_with_phase_least_squares(
    flux_values: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    """Fit ``x0 + xA cos(2*pi*phi + phase)`` by linear least-squares in cos/sin basis."""
    flux_values = np.asarray(flux_values, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    theta = 2.0 * np.pi * flux_values
    c = np.cos(theta)
    s = np.sin(theta)
    X = np.column_stack([np.ones_like(theta), c, s])
    beta, *_ = np.linalg.lstsq(X, values, rcond=None)
    x0 = float(beta[0])
    a = float(beta[1])  # cos coefficient
    b = float(beta[2])  # sin coefficient
    model = x0 + a * c + b * s

    # a*cos(theta) + b*sin(theta) = xA*cos(theta + phase)
    # => a = xA*cos(phase), b = -xA*sin(phase)
    xA = float(np.hypot(a, b))
    phase = float(np.arctan2(-b, a))
    return model, x0, xA, phase


def _safe_reciprocal(values: np.ndarray, eps: float = 5e-2) -> np.ndarray:
    """Stable reciprocal with a dispersive-validity floor near resonances."""
    values = np.asarray(values, dtype=float)
    signs = np.where(values >= 0.0, 1.0, -1.0)
    denom = np.where(np.abs(values) < eps, signs * eps, values)
    return 1.0 / denom


def build_physical_templates(
    flux_values: np.ndarray,
    *,
    wc0: float,
    A: float,
    ham_kwargs: dict[str, float | int],
) -> dict[str, np.ndarray]:
    """Compute physically-motivated dispersive templates vs flux."""
    flux_values = np.asarray(flux_values, dtype=float).ravel()
    w1_bare = float(ham_kwargs["w_1"])
    w2_bare = float(ham_kwargs["w_2"])
    alpha_c = float(ham_kwargs["alpha_c"])
    g1c = float(ham_kwargs["g_1c"])
    g2c = float(ham_kwargs["g_2c"])

    wc = np.asarray(coupler_frequency(wc0, A, flux_values), dtype=float).ravel()
    d1 = w1_bare - wc
    d2 = w2_bare - wc
    d1c = d1 + alpha_c
    d2c = d2 + alpha_c

    inv_d1 = _safe_reciprocal(d1)
    inv_d2 = _safe_reciprocal(d2)
    inv_d1c = _safe_reciprocal(d1c)
    inv_d2c = _safe_reciprocal(d2c)

    # Standard dispersive-like templates for coupler-mediated effective interactions.
    j_med = 0.5 * g1c * g2c * (inv_d1 + inv_d2 - inv_d1c - inv_d2c)
    chi1 = g1c * g1c * (inv_d1 - inv_d1c)
    chi2 = g2c * g2c * (inv_d2 - inv_d2c)

    return {
        "wc": wc,
        "inv_d1": inv_d1,
        "inv_d2": inv_d2,
        "inv_d1c": inv_d1c,
        "inv_d2c": inv_d2c,
        "j_med": j_med,
        "chi1": chi1,
        "chi2": chi2,
    }


def fit_physical_metric_models(
    flux_values: np.ndarray,
    true_metrics: dict[str, np.ndarray],
    *,
    wc0: float,
    A: float,
    ham_kwargs: dict[str, float | int],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Fit metric-specific linear models on dispersive template features."""
    templates = build_physical_templates(
        flux_values,
        wc0=wc0,
        A=A,
        ham_kwargs=ham_kwargs,
    )

    one = np.ones_like(templates["wc"])
    j_med = templates["j_med"]
    chi1 = templates["chi1"]
    chi2 = templates["chi2"]
    inv_d1 = templates["inv_d1"]
    inv_d2 = templates["inv_d2"]
    inv_d1c = templates["inv_d1c"]
    inv_d2c = templates["inv_d2c"]

    design_by_metric = {
        # Detuning-rational Lamb-shift style basis (up to second reciprocal order).
        "w1": np.column_stack(
            [
                one,
                templates["wc"],
                inv_d1,
                inv_d1c,
                inv_d1 * inv_d1,
                inv_d1c * inv_d1c,
                j_med * j_med,
            ]
        ),
        "w2": np.column_stack(
            [
                one,
                templates["wc"],
                inv_d2,
                inv_d2c,
                inv_d2 * inv_d2,
                inv_d2c * inv_d2c,
                j_med * j_med,
            ]
        ),
        # Exchange from mediated coupling plus detuning corrections.
        "J": np.column_stack(
            [
                one,
                j_med,
                inv_d1 + inv_d2,
                inv_d1c + inv_d2c,
                j_med * (inv_d1 + inv_d2),
            ]
        ),
        # ZZ from fourth-order-like products and mediated exchange powers.
        "zeta": np.column_stack(
            [
                one,
                chi1 * chi2,
                j_med * j_med,
                inv_d1 * inv_d2,
                inv_d1c * inv_d2c,
                j_med * (inv_d1 - inv_d2),
            ]
        ),
    }

    model_metrics: dict[str, np.ndarray] = {}
    coeffs: dict[str, np.ndarray] = {}
    for name, X in design_by_metric.items():
        y = np.asarray(true_metrics[name], dtype=float).ravel()
        scales = np.linalg.norm(X, axis=0)
        scales = np.where(scales > 0.0, scales, 1.0)
        X_scaled = X / scales
        beta_scaled, *_ = np.linalg.lstsq(X_scaled, y, rcond=None)
        beta = beta_scaled / scales
        model_metrics[name] = X @ beta
        coeffs[name] = beta

    return model_metrics, coeffs


def plot_true_vs_models(
    flux_values: np.ndarray,
    true_metrics: dict[str, np.ndarray],
    cosine_meanmax_metrics: dict[str, np.ndarray],
    cosine_phase_metrics: dict[str, np.ndarray],
    physical_metrics: dict[str, np.ndarray],
    *,
    outfile: str,
) -> None:
    """Plot true metric curves and surrogate models."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    names = ("w1", "w2", "J", "zeta")
    labels = {
        "w1": r"$w_1$ (GHz)",
        "w2": r"$w_2$ (GHz)",
        "J": r"$J$ (GHz)",
        "zeta": r"$\zeta$ (GHz)",
    }

    for ax, name in zip(axes.ravel(), names):
        ax.plot(flux_values, true_metrics[name], label="Model-2 metric", color="C0", linewidth=1.8)
        ax.plot(
            flux_values,
            cosine_meanmax_metrics[name],
            label=r"Cos (mean/max): $x_0 + x_A \cos(2\pi \phi)$",
            color="C3",
            linestyle="--",
            linewidth=1.6,
        )
        ax.plot(
            flux_values,
            cosine_phase_metrics[name],
            label=r"Cos (LS+phase): $x_0 + x_A \cos(2\pi \phi + \varphi)$",
            color="C2",
            linestyle="-.",
            linewidth=1.5,
        )
        ax.plot(
            flux_values,
            physical_metrics[name],
            label="Dispersive model",
            color="C4",
            linestyle=":",
            linewidth=1.8,
        )
        ax.set_ylabel(labels[name])
        if name == "zeta":
            ax.axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize="small")

    axes[1, 0].set_xlabel(r"Flux bias ($\Phi / \Phi_0$)")
    axes[1, 1].set_xlabel(r"Flux bias ($\Phi / \Phi_0$)")
    fig.suptitle("Full-range model-2 metrics vs cosine and dispersive surrogates")
    fig.tight_layout()
    plt.savefig(outfile, format="pdf")
    plt.close(fig)


def summarize_fit(name: str, true_values: np.ndarray, model_values: np.ndarray) -> str:
    """Return compact RMSE / max_abs error summary."""
    err = np.asarray(model_values, dtype=float) - np.asarray(true_values, dtype=float)
    rmse = float(np.sqrt(np.mean(err * err)))
    emax = float(np.max(np.abs(err)))
    return f"{name:>5}: rmse={rmse:.3e}, max_abs={emax:.3e}"


def main() -> None:
    wc0 = 6.0
    A = 1.0
    ham_kwargs = {
        "w_1": 5.0,
        "w_2": 5.2,
        "alpha_1": -0.3,
        "alpha_c": -0.3,
        "alpha_2": -0.3,
        "g_1c": 0.08,
        "g_2c": 0.08,
        "nlevels_qubit": 2,
        "nlevels_coupler": 2,
    }

    flux = np.linspace(0.0, 1.0, 401)
    H2 = three_mode_hamiltonian_stack_vs_flux(
        flux,
        wc0=wc0,
        A=A,
        ham_kwargs=ham_kwargs,
    )
    H2_eff = build_dressed_effective_computational_stack(
        H2,
        nlevels_qubit=int(ham_kwargs["nlevels_qubit"]),
        nlevels_coupler=int(ham_kwargs["nlevels_coupler"]),
    )
    true_metrics = extract_model1_parameters_from_4x4_stack(H2_eff)

    cosine_metrics_meanmax: dict[str, np.ndarray] = {}
    coeffs_meanmax: dict[str, tuple[float, float]] = {}
    cosine_metrics_phase: dict[str, np.ndarray] = {}
    coeffs_phase: dict[str, tuple[float, float, float]] = {}
    for name in ("w1", "w2", "J", "zeta"):
        model_mm, x0_mm, xA_mm = cosine_from_mean_and_max(flux, true_metrics[name])
        model_ph, x0_ph, xA_ph, phase_ph = cosine_with_phase_least_squares(flux, true_metrics[name])
        cosine_metrics_meanmax[name] = model_mm
        coeffs_meanmax[name] = (x0_mm, xA_mm)
        cosine_metrics_phase[name] = model_ph
        coeffs_phase[name] = (x0_ph, xA_ph, phase_ph)

    physical_metrics, physical_coeffs = fit_physical_metric_models(
        flux,
        true_metrics,
        wc0=wc0,
        A=A,
        ham_kwargs=ham_kwargs,
    )

    outdir = Path(__file__).resolve().parent
    outfile = outdir / "model2_metrics_surrogates_vs_true_full_range.pdf"
    plot_true_vs_models(
        flux,
        true_metrics,
        cosine_metrics_meanmax,
        cosine_metrics_phase,
        physical_metrics,
        outfile=str(outfile),
    )

    print("Cosine surrogate coefficients from mean/max:")
    for name in ("w1", "w2", "J", "zeta"):
        x0, xA = coeffs_meanmax[name]
        print(f"  {name:>5}: {name}_0={x0:.8f}, {name}_A={xA:.8f}")

    print("\nCosine surrogate coefficients from least-squares with phase:")
    for name in ("w1", "w2", "J", "zeta"):
        x0, xA, phase = coeffs_phase[name]
        print(f"  {name:>5}: {name}_0={x0:.8f}, {name}_A={xA:.8f}, phase={phase:.8f} rad")

    print("\nDispersive-model linear coefficients:")
    for name in ("w1", "w2", "J", "zeta"):
        print(f"  {name:>5}: {physical_coeffs[name]}")

    print("\nFit error summary:")
    for name in ("w1", "w2", "J", "zeta"):
        mm = summarize_fit(name, true_metrics[name], cosine_metrics_meanmax[name])
        ph = summarize_fit(name, true_metrics[name], cosine_metrics_phase[name])
        pm = summarize_fit(name, true_metrics[name], physical_metrics[name])
        print(f"  {mm}    | phase-fit {ph}    | phys-fit {pm}")

    print("\nWrote:")
    print(f"  {outfile}")


if __name__ == "__main__":
    main()
