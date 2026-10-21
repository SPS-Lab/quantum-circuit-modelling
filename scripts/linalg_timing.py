"""Compare CPU/GPU expm, matrix multiplication, and general linear solve.

Run from the repo root, for example:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python scripts/linalg_timing.py

Hamiltonian construction, warm-up, host/device transfers, and numerical checks
are outside the measured calls. GPU times include waiting for completion.
All operations use complex128 matrices of the circuit exponent's dimension.
The solve uses a full matrix of right-hand sides, as needed in Pade evaluation;
these component timings are diagnostics, not an exact decomposition of expm.
"""

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys
import time

import cupy as cp
from cupyx.scipy.linalg import expm as cuexpm
import numpy as np
from scipy.linalg import expm, solve


def create_hamiltonian_exp(qubit_dim=17):
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from models.circuit import build_circuit_model_stack
    from study_config import load_study_config

    target_flux = 0.23333333333333334

    config = load_study_config(
        system_params_path=_REPO_ROOT / "params/system_params.json",
        study_params_path=_REPO_ROOT / "params/benchmark_params.json",
    )
    circuit = config.static_benchmark.circuit_model
    circuit = replace(
        circuit,
        hilbert_truncation=replace(
            circuit.hilbert_truncation,
            q0_truncated_dim=qubit_dim,
            q1_truncated_dim=qubit_dim,
        ),
    )
    sweep_target = config.static_benchmark.flux_control.sweep_target

    hamiltonians = build_circuit_model_stack(
        flux_values=np.asarray([target_flux]),
        system_params=config.system,
        circuit_config=circuit,
        sweep_target=sweep_target,
    ).hamiltonian_stack

    hamiltonian = hamiltonians[0]
    dt = 0.5
    return (-1.0j * 2.0 * np.pi * dt) * hamiltonian


def time_operation(function, arguments, repeats, stream=None):
    samples = []
    result = None
    for _ in range(repeats):
        result = None  # Release the previous output outside the timed call.
        if stream is not None:
            stream.synchronize()
        started = time.perf_counter()
        result = function(*arguments)
        if stream is not None:
            stream.synchronize()
        samples.append(time.perf_counter() - started)
    return result, np.asarray(samples)


def compare_operation(
    label, cpu_function, gpu_function, cpu_args, gpu_args, repeats, stream
):
    print(f"Warming up CPU and GPU {label}...", flush=True)
    cpu_function(*cpu_args)
    gpu_function(*gpu_args)
    stream.synchronize()

    res_cpu, cpu_times = time_operation(cpu_function, cpu_args, repeats)
    res_gpu, gpu_times = time_operation(gpu_function, gpu_args, repeats, stream)
    for name, samples in (("CPU", cpu_times), ("GPU", gpu_times)):
        print(
            f"{name} {label}: median={np.median(samples):.6f} s, "
            f"min={samples.min():.6f} s, max={samples.max():.6f} s "
            f"({repeats} repeats)",
            flush=True,
        )
    print(
        f"{label} CPU/GPU median ratio: {np.median(cpu_times) / np.median(gpu_times):.3f} "
        "(>1 means GPU faster)",
        flush=True,
    )

    # Transfers and checks are deliberately outside the timings.
    res_gpu = cp.asnumpy(res_gpu)
    if not (np.isfinite(res_cpu).all() and np.isfinite(res_gpu).all()):
        raise RuntimeError(f"{label} returned non-finite values")
    denominator = max(np.linalg.norm(res_cpu, "fro"), np.finfo(float).tiny)
    relative_error = np.linalg.norm(res_gpu - res_cpu, "fro") / denominator
    print(f"{label} CPU/GPU relative Frobenius error: {relative_error:.3e}", flush=True)
    if not np.isfinite(relative_error) or relative_error > 1e-8:
        raise RuntimeError(f"{label} CPU/GPU agreement check failed (tolerance 1e-8)")
    return res_cpu, res_gpu


def general_solve(a, b):
    # Match the GPU's general dense solve, without an extra CPU finiteness scan.
    return solve(a, b, assume_a="gen", check_finite=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubit-dim", type=int, default=17)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.qubit_dim < 2 or args.repeats < 1:
        parser.error("qubit-dim must be >= 2 and repeats must be >= 1")

    exponent = create_hamiltonian_exp(args.qubit_dim)
    print(f"Exponent: shape={exponent.shape}, dtype={exponent.dtype}", flush=True)
    print(
        "CPU thread environment:",
        {
            name: os.environ.get(name, "unset")
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        flush=True,
    )
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        print("Install threadpoolctl to report actual BLAS thread counts.", flush=True)
    else:
        for pool in threadpool_info():
            if pool["user_api"] == "blas":
                print(
                    f"BLAS: {pool['prefix']}, threads={pool['num_threads']}", flush=True
                )

    exponent_gpu = cp.asarray(exponent)
    stream = cp.cuda.get_current_stream()
    stream.synchronize()
    res_cpu, res_gpu = compare_operation(
        "expm", expm, cuexpm, (exponent,), (exponent_gpu,), args.repeats, stream
    )
    identity = np.eye(exponent.shape[0], dtype=exponent.dtype)
    for name, result in (("SciPy", res_cpu), ("CuPy", res_gpu)):
        error = np.linalg.norm(result.conj().T @ result - identity, "fro") / np.sqrt(
            result.shape[0]
        )
        print(f"{name} relative unitarity error: {error:.3e}", flush=True)
        if not np.isfinite(error) or error > 1e-8:
            raise RuntimeError(f"{name} unitarity check failed (tolerance 1e-8)")
    del res_cpu, res_gpu, result

    compare_operation(
        "matmul",
        np.matmul,
        cp.matmul,
        (exponent, exponent),
        (exponent_gpu, exponent_gpu),
        args.repeats,
        stream,
    )

    # Scaling keeps this diagnostic system well-conditioned. I +/- a scaled
    # skew-Hermitian matrix mimics the general matrix solve in Pade evaluation.
    scaled = exponent / max(np.linalg.norm(exponent, 1), 1.0)
    a = identity - scaled
    b = identity + scaled
    a_gpu, b_gpu = cp.asarray(a), cp.asarray(b)
    stream.synchronize()
    res_cpu, res_gpu = compare_operation(
        "solve (matrix RHS)",
        general_solve,
        cp.linalg.solve,
        (a, b),
        (a_gpu, b_gpu),
        args.repeats,
        stream,
    )
    for name, result in (("SciPy", res_cpu), ("CuPy", res_gpu)):
        residual = np.linalg.norm(a @ result - b, "fro") / np.linalg.norm(b, "fro")
        print(f"{name} relative solve residual: {residual:.3e}", flush=True)
        if not np.isfinite(residual) or residual > 1e-8:
            raise RuntimeError(f"{name} solve residual check failed (tolerance 1e-8)")


if __name__ == "__main__":
    main()
