import numpy as np
import cupy as cp
from scipy.linalg import expm
from cupyx.scipy.linalg import expm as cuexpm
from pathlib import Path
import sys
from dataclasses import replace
import time


def create_hamiltonian_exp():
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from models.circuit import build_circuit_model_stack
    from study_config import load_study_config

    QUBIT_DIM = 17

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
            q0_truncated_dim=QUBIT_DIM,
            q1_truncated_dim=QUBIT_DIM,
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


# exponent = create_hamiltonian_exp()
# exponent = np.random.rand(1300,1300) #scipy: 0.873s, cupyx: 1.752s
exponent = np.random.rand(1400, 1400)  # scipy: 1.188s, cupyx: 2.106s
# exponent = np.random.rand(1500,1500) #Overflow in scipy, cupyx ok!
# exponent = np.random.rand(1734,1734) #Overflow in scipy, cupyx ok!

print("exponent shape: ", exponent.shape)

started = time.perf_counter()
res_cpu = expm(exponent)
print(f"scipy.linalg.expm time: {time.perf_counter() - started:.3f} s", flush=True)

exponent = cp.array(exponent)
started = time.perf_counter()
res_gpu = cuexpm(exponent)
cp.cuda.Stream.null.synchronize()
print(
    f"cupyx.scipy.linalg.expm time: {time.perf_counter() - started:.3f} s", flush=True
)


print("res_cpu shape:", res_cpu.shape)
print("res_gpu shape:", cp.asnumpy(res_gpu).shape)
