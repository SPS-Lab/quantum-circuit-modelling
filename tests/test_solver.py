from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from comparison.cz import _simulate_piecewise_constant_scipy
from toolkit.observables import extract_cz_observables
from toolkit.solver import propagate_piecewise_constant
from toolkit.pulses import ramp_hold_ramp_flux_pulse
import toolkit.solver as solver_module


def test_piecewise_propagation_matches_analytic_rotations() -> None:
    # Noncommuting X then Z rotations on an irregular time grid verify the
    # interval convention, multiplication order, and GHz/ns conversion.
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.diag([1, -1]).astype(complex)
    hamiltonians = np.stack([0.25 * x, 0.5 * z, 99 * x])
    times = np.array([0.0, 0.5, 0.75])
    initial = np.array([[1.0, 1 / np.sqrt(2)], [0.0, 1j / np.sqrt(2)]])
    before = initial.copy()
    ux = (np.eye(2) - 1j * x) / np.sqrt(2)
    uz = np.diag(np.exp(-1j * np.pi / 4 * np.array([1, -1])))
    expected = np.stack([initial, ux @ initial, uz @ ux @ initial])

    actual = propagate_piecewise_constant(hamiltonians, times, initial)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(np.sum(np.abs(actual) ** 2, axis=1), 1.0)
    np.testing.assert_array_equal(initial, before)
    np.testing.assert_array_equal(hamiltonians, np.stack([0.25 * x, 0.5 * z, 99 * x]))

    # Recording one row must still propagate through the omitted basis row.
    projected = propagate_piecewise_constant(
        hamiltonians, times, initial, output_indices=np.array([0])
    )
    np.testing.assert_allclose(projected, expected[:, [0], :], atol=1e-14)
    reordered = propagate_piecewise_constant(
        hamiltonians, times, initial, output_indices=np.array([1, 0])
    )
    np.testing.assert_allclose(reordered, expected[:, [1, 0], :], atol=1e-14)


def test_single_time_returns_initial_state() -> None:
    initial = np.eye(2, dtype=complex)
    result = propagate_piecewise_constant(np.zeros((1, 2, 2)), np.array([3.0]), initial)
    np.testing.assert_array_equal(result, initial[None, :, :])


@pytest.mark.parametrize(
    "hold, dt, target",
    [
        (3.25, 0.5, 0.2),
        (0.0, 0.5, 0.2),
        (2.0, 0.1, 0.2),
        (3.25, 0.5, 0.0),
        (0.0, 10.0, 0.2),
    ],
)
def test_compressed_pulse_preserves_grid_and_interval_flux(hold, dt, target) -> None:
    kwargs = dict(
        ramp_time_ns=1.0, hold_time_ns=hold, dt_ns=dt, idle_flux=0.0, target_flux=target
    )
    full_times, full_flux = ramp_hold_ramp_flux_pulse(**kwargs)
    times, flux, counts = ramp_hold_ramp_flux_pulse(**kwargs, compress=True)
    np.testing.assert_array_equal(times, full_times)
    np.testing.assert_array_equal(np.repeat(flux, counts), full_flux[:-1])
    assert np.issubdtype(counts.dtype, np.integer)
    assert np.all(counts > 0)
    assert counts.sum() == times.size - 1
    # Also preserve the trajectory for non-binary dt (0.1), where subtraction
    # of adjacent grid values gives several distinct floating-point lengths.
    x = np.array([[0.0, 0.1], [0.1, 0.0]])
    z = np.diag([1.0, -1.0])
    full_h = x[None, :, :] + full_flux[:, None, None] * z
    compressed_h = x[None, :, :] + flux[:, None, None] * z
    expected = propagate_piecewise_constant(full_h, times, np.eye(2))
    actual = propagate_piecewise_constant(compressed_h, times, np.eye(2), counts=counts)
    np.testing.assert_array_equal(actual, expected)


def test_compressed_propagation_reuses_expm_and_preserves_short_final_step(
    monkeypatch,
) -> None:
    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.25])
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])
    segments = np.stack([0.25 * x, 0.5 * z])
    counts = np.array([2, 3])
    full = np.concatenate((np.repeat(segments, counts, axis=0), segments[-1:]))
    initial = np.eye(2, dtype=complex)
    expected = propagate_piecewise_constant(
        full, times, initial, output_indices=np.array([1, 0])
    )
    calls = []
    original_expm = solver_module.expm

    def count_expm(exponent):
        calls.append(exponent.copy())
        return original_expm(exponent)

    monkeypatch.setattr(solver_module, "expm", count_expm)
    actual = propagate_piecewise_constant(
        segments, times, initial, counts=counts, output_indices=np.array([1, 0])
    )
    np.testing.assert_array_equal(actual, expected)
    assert len(calls) == 3  # X at .5 ns, Z at .5 ns, and Z at .25 ns.


def test_default_cz_pulse_needs_nine_exponentials(monkeypatch) -> None:
    times, flux, counts = ramp_hold_ramp_flux_pulse(
        ramp_time_ns=2.0,
        hold_time_ns=237.9595316591736,
        dt_ns=0.5,
        idle_flux=0.0,
        target_flux=14 / 60,
        compress=True,
    )
    hamiltonians = flux[:, None, None] * np.diag([1.0, -1.0])[None, :, :]
    calls = []
    original_expm = solver_module.expm

    def count_expm(exponent):
        calls.append(1)
        return original_expm(exponent)

    monkeypatch.setattr(solver_module, "expm", count_expm)
    amplitudes = propagate_piecewise_constant(
        hamiltonians, times, np.eye(2), counts=counts
    )
    assert amplitudes.shape == (485, 2, 2)
    assert len(calls) == 9
    assert counts.max() == 476


@pytest.mark.parametrize(
    "counts",
    [
        np.array([0, 2]),
        np.array([-1, 3]),
        np.array([1.0, 1.0]),
        np.array([True, True]),
        np.array([1]),
        np.array([[1, 1]]),
        np.array([1, 2]),
    ],
)
def test_invalid_segment_counts(counts) -> None:
    with pytest.raises(ValueError, match="counts"):
        propagate_piecewise_constant(
            np.zeros((2, 2, 2)),
            np.array([0.0, 0.5, 1.0]),
            np.eye(2),
            counts=counts,
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"hamiltonians_ghz": np.zeros((2, 2))}, "hamiltonians_ghz"),
        ({"hamiltonians_ghz": np.zeros((2, 2, 3))}, "hamiltonians_ghz"),
        ({"times_ns": np.array([0.0])}, "times_ns"),
        ({"times_ns": np.array([0.0, 0.0])}, "times_ns"),
        ({"times_ns": np.array([1.0, 0.0])}, "times_ns"),
        ({"times_ns": np.array([0.0, np.nan])}, "times_ns"),
        ({"times_ns": np.array([[0.0, 1.0]])}, "times_ns"),
        ({"initial_states": np.ones(2)}, "initial_states"),
        ({"initial_states": np.ones((3, 1))}, "initial_states"),
        ({"output_indices": np.array([0.5])}, "output_indices"),
        ({"output_indices": np.array([-1])}, "output_indices"),
        ({"output_indices": np.array([2])}, "output_indices"),
    ],
)
def test_invalid_propagation_inputs(kwargs: dict, message: str) -> None:
    inputs = dict(
        hamiltonians_ghz=np.zeros((2, 2, 2)),
        times_ns=np.array([0.0, 1.0]),
        initial_states=np.eye(2),
    )
    inputs.update(kwargs)
    with pytest.raises(ValueError, match=message):
        propagate_piecewise_constant(**inputs)


def test_cz_adapter_preserves_leakage_and_monitored_population() -> None:
    hamiltonian = np.zeros((5, 5))
    hamiltonian[3, 4] = hamiltonian[4, 3] = 0.25
    result = _simulate_piecewise_constant_scipy(
        np.stack([hamiltonian] * 3),
        np.array([0.0, 0.5, 1.0]),
        np.arange(4),
        monitor_indices=np.array([4]),
    )
    expected_leakage = np.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(result.leakage_11, expected_leakage, atol=1e-14)
    np.testing.assert_allclose(
        result.monitor_population_11, expected_leakage, atol=1e-14
    )
    np.testing.assert_allclose(
        result.populations_11[:, 3], 1 - expected_leakage, atol=1e-14
    )


def test_cz_analysis_extracts_conditional_phase() -> None:
    amplitudes = np.stack([np.eye(4, dtype=complex)] * 2)
    amplitudes[1, 3, 3] = np.exp(-0.7j)
    result = extract_cz_observables(amplitudes)
    np.testing.assert_allclose(result.conditional_phase, [0.0, -0.7])
    np.testing.assert_allclose(result.leakage_11, 0.0, atol=1e-14)
    np.testing.assert_array_equal(result.monitor_population_11, np.zeros(2))
