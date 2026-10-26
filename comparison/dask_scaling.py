"""Strong scaling of a fixed circuit pulse with threaded and process Dask workers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
import json
import os
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from runtime_utils import log_progress
from study_config import StudyConfig
from toolkit.pulses import ramp_hold_ramp_flux_pulse
from toolkit.simulation import simulate_circuit


@dataclass(frozen=True)
class DaskScalingSettings:
    qubit_dim: int = 8
    parallelism: tuple[int, ...] = (1, 2, 4, 8)
    repeats: int = 5
    warmups: int = 1
    blas_threads: int = 1
    seed: int = 0
    target_flux: float = 0.23333333333333334
    ramp_time_ns: float = 2.0
    hold_time_ns: float = 237.9595316591736
    dt_ns: float = 0.5

    def __post_init__(self):
        for name, minimum in (
            ("qubit_dim", 2),
            ("repeats", 1),
            ("warmups", 1),
            ("blas_threads", 1),
            ("seed", 0),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        values = tuple(self.parallelism)
        if (
            not values
            or values[0] != 1
            or any(
                isinstance(p, bool) or not isinstance(p, (int, np.integer)) or p < 1
                for p in values
            )
            or list(values) != sorted(set(values))
        ):
            raise ValueError(
                "parallelism must be strictly increasing positive integers starting with 1"
            )
        object.__setattr__(self, "parallelism", values)
        if not all(
            np.isfinite(getattr(self, name))
            for name in ("target_flux", "ramp_time_ns", "hold_time_ns", "dt_ns")
        ):
            raise ValueError("pulse settings must be finite")
        if self.ramp_time_ns <= 0 or self.hold_time_ns < 0 or self.dt_ns <= 0:
            raise ValueError(
                "ramp_time_ns and dt_ns must be positive; hold_time_ns must be nonnegative"
            )


@dataclass(frozen=True)
class DaskScalingResult:
    """Raw samples use axes (mode, parallelism, repeat); serial samples use (repeat,)."""

    modes: np.ndarray
    parallelism: np.ndarray
    repeats: int
    warmups: int
    blas_threads: int
    times_ns: np.ndarray
    flux_values: np.ndarray
    counts: np.ndarray
    reference_amplitudes: np.ndarray
    build_samples_s: np.ndarray
    preparation_samples_s: np.ndarray
    state_propagation_samples_s: np.ndarray
    serial_build_samples_s: np.ndarray
    serial_preparation_samples_s: np.ndarray
    serial_state_propagation_samples_s: np.ndarray
    cluster_startup_s: np.ndarray
    max_amplitude_difference: np.ndarray
    execution_order: np.ndarray
    settings_json: str
    environment_json: str

    @property
    def propagation_samples_s(self):
        return self.preparation_samples_s + self.state_propagation_samples_s

    @property
    def preparation_speedup(self):
        medians = np.median(self.preparation_samples_s, axis=2)
        return medians[:, :1] / medians

    @property
    def propagation_speedup(self):
        medians = np.median(self.propagation_samples_s, axis=2)
        return medians[:, :1] / medians

    @property
    def preparation_efficiency(self):
        return self.preparation_speedup / self.parallelism[None, :]


def _configure_worker(blas_threads):
    # Keep the controller alive for the worker's lifetime. NumPy/SciPy have
    # already been imported when this module is loaded in each worker.
    global _worker_thread_limiter
    _worker_thread_limiter = threadpool_limits(limits=blas_threads)
    return {
        "pid": os.getpid(),
        "affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "threadpools": threadpool_info(),
    }


def run_dask_scaling_benchmark(
    config: StudyConfig,
    settings: DaskScalingSettings,
    *,
    dashboard_address: str = "127.0.0.1:8787",
) -> DaskScalingResult:
    """Measure a fixed pulse, changing only Dask concurrency.

    Threads use one worker with p threads; processes use p workers with one
    thread each. Each cluster is reused for full-workload warmups and measured
    repeats, then closed. Cluster groups run in seeded shuffled order. Builds
    are serial and reported separately. Preparation includes scatter/gather;
    startup, teardown, warmups, and numerical checks are outside timings.
    """
    from dask.distributed import Client

    circuit = config.static_benchmark.circuit_model
    max_dim = min(
        2 * circuit.transmon_charge_basis.q0_ncut + 1,
        2 * circuit.transmon_charge_basis.q1_ncut + 1,
    )
    if settings.qubit_dim > max_dim:
        raise ValueError(f"qubit_dim exceeds charge-basis dimension {max_dim}")
    circuit = replace(
        circuit,
        hilbert_truncation=replace(
            circuit.hilbert_truncation,
            q0_truncated_dim=settings.qubit_dim,
            q1_truncated_dim=settings.qubit_dim,
        ),
    )
    sweep_target = config.static_benchmark.flux_control.sweep_target
    if sweep_target not in ("q0", "q1"):
        raise ValueError("This CZ pulse benchmark requires a qubit flux sweep target")
    idle_flux = getattr(config.system, sweep_target).flux
    times, flux, counts = ramp_hold_ramp_flux_pulse(
        ramp_time_ns=settings.ramp_time_ns,
        hold_time_ns=settings.hold_time_ns,
        dt_ns=settings.dt_ns,
        idle_flux=idle_flux,
        target_flux=settings.target_flux,
        compress=True,
    )
    kwargs = dict(
        system_params=config.system,
        circuit_config=circuit,
        sweep_target=sweep_target,
        times_ns=times,
        flux_values=flux,
        counts=counts,
    )
    modes = np.array(["threads", "processes"])
    shape = (2, len(settings.parallelism), settings.repeats)
    build, preparation, propagation = (np.empty(shape) for _ in range(3))
    differences = np.empty(shape)
    serial_build, serial_preparation, serial_propagation = (
        np.empty(settings.repeats) for _ in range(3)
    )
    startup = np.empty(shape[:2])
    order = np.array([(m, p) for m in range(2) for p in range(shape[1])])
    np.random.default_rng(settings.seed).shuffle(order)
    environment = {
        "packages": {
            name: version(name)
            for name in (
                "numpy",
                "scipy",
                "dask",
                "distributed",
                "threadpoolctl",
                "scqubits",
            )
        },
        "clusters": {},
    }
    with threadpool_limits(limits=settings.blas_threads):
        environment["parent_threadpools"] = threadpool_info()
        log_progress("Dask scaling: warming serial reference")
        for _ in range(settings.warmups):
            reference = simulate_circuit(**kwargs)
        reference_amplitudes = reference.amplitudes

        def check(result):
            np.testing.assert_allclose(
                result.amplitudes,
                reference_amplitudes,
                rtol=1e-10,
                atol=1e-10,
                err_msg="Dask scaling changed the simulated trajectory",
            )
            return float(np.max(np.abs(result.amplitudes - reference_amplitudes)))

        for repeat in range(settings.repeats):
            log_progress(f"Dask scaling: serial repeat {repeat + 1}/{settings.repeats}")
            sample = simulate_circuit(**kwargs)
            check(sample)
            serial_build[repeat] = sample.build_runtime_s
            serial_preparation[repeat] = sample.preparation_runtime_s
            serial_propagation[repeat] = sample.state_propagation_runtime_s

        for mode_index, parallel_index in order:
            p = settings.parallelism[parallel_index]
            mode = str(modes[mode_index])
            label = f"Dask scaling: {mode}, p={p}"
            log_progress(f"{label}: starting cluster")
            started = perf_counter()
            with Client(
                processes=mode == "processes",
                n_workers=p if mode == "processes" else 1,
                threads_per_worker=1 if mode == "processes" else p,
                dashboard_address=dashboard_address,
            ) as client:
                startup[mode_index, parallel_index] = perf_counter() - started
                workers = client.run(_configure_worker, settings.blas_threads)
                scheduler = client.scheduler_info()
                environment["clusters"][f"{mode}_{p}"] = {
                    "worker_environment": workers,
                    "workers": {
                        address: {
                            key: info[key] for key in ("nthreads", "memory_limit")
                        }
                        for address, info in scheduler["workers"].items()
                    },
                }
                host = dashboard_address.rsplit(":", 1)[0] or "127.0.0.1"
                if host in ("0.0.0.0", "[::]"):
                    host = "127.0.0.1"
                log_progress(
                    f"{label}: dashboard http://{host}:{scheduler['services']['dashboard']}/status"
                )
                for warmup in range(settings.warmups):
                    log_progress(f"{label}: warmup {warmup + 1}/{settings.warmups}")
                    check(simulate_circuit(**kwargs, client=client))
                for repeat in range(settings.repeats):
                    log_progress(f"{label}: repeat {repeat + 1}/{settings.repeats}")
                    sample = simulate_circuit(**kwargs, client=client)
                    index = (mode_index, parallel_index, repeat)
                    differences[index] = check(sample)
                    build[index] = sample.build_runtime_s
                    preparation[index] = sample.preparation_runtime_s
                    propagation[index] = sample.state_propagation_runtime_s
                    log_progress(
                        f"{label}: preparation={preparation[index]:.6f} s, "
                        f"propagation={sample.propagation_runtime_s:.6f} s"
                    )
    return DaskScalingResult(
        modes=modes,
        parallelism=np.array(settings.parallelism),
        repeats=settings.repeats,
        warmups=settings.warmups,
        blas_threads=settings.blas_threads,
        times_ns=times,
        flux_values=flux,
        counts=counts,
        reference_amplitudes=reference_amplitudes,
        build_samples_s=build,
        preparation_samples_s=preparation,
        state_propagation_samples_s=propagation,
        serial_build_samples_s=serial_build,
        serial_preparation_samples_s=serial_preparation,
        serial_state_propagation_samples_s=serial_propagation,
        cluster_startup_s=startup,
        max_amplitude_difference=differences,
        execution_order=order,
        settings_json=json.dumps(
            {
                "benchmark": asdict(settings),
                "system": asdict(config.system),
                "circuit_config": asdict(circuit),
                "sweep_target": sweep_target,
                "idle_flux": idle_flux,
            },
            indent=2,
        ),
        environment_json=json.dumps(environment, indent=2),
    )
