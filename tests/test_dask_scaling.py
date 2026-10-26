from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
from threadpoolctl import threadpool_info

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark_results_io import load_result_hdf5
from comparison.dask_scaling import (
    DaskScalingResult,
    DaskScalingSettings,
    run_dask_scaling_benchmark,
)
from study_config import load_study_config


@pytest.fixture(scope="module")
def scaling_result():
    pytest.importorskip("distributed")
    config = load_study_config(
        system_params_path=_ROOT / "params/system_params.json",
        study_params_path=_ROOT / "params/benchmark_params.json",
    )
    circuit = config.static_benchmark.circuit_model
    circuit = replace(
        circuit,
        hilbert_truncation=replace(circuit.hilbert_truncation, c_truncated_dim=2),
    )
    config = replace(
        config, static_benchmark=replace(config.static_benchmark, circuit_model=circuit)
    )
    settings = DaskScalingSettings(
        qubit_dim=2,
        parallelism=(1, 2),
        repeats=2,
        ramp_time_ns=0.5,
        hold_time_ns=0.75,
        dt_ns=0.5,
    )
    original = [(p["filepath"], p["num_threads"]) for p in threadpool_info()]
    result = run_dask_scaling_benchmark(
        config, settings, dashboard_address="127.0.0.1:0"
    )
    assert [(p["filepath"], p["num_threads"]) for p in threadpool_info()] == original
    return result


def test_real_thread_and_process_scaling(scaling_result):
    result = scaling_result
    for samples in (
        result.build_samples_s,
        result.preparation_samples_s,
        result.state_propagation_samples_s,
    ):
        assert samples.shape == (2, 2, 2)
        assert np.all(np.isfinite(samples))
        assert np.all(samples > 0)
    assert result.cluster_startup_s.shape == (2, 2)
    assert np.all(result.cluster_startup_s > 0)
    assert result.counts.sum() == len(result.times_ns) - 1
    assert result.reference_amplitudes.shape == (len(result.times_ns), 4, 4)
    assert result.max_amplitude_difference.max() < 1e-10
    np.testing.assert_allclose(result.preparation_speedup[:, 0], 1)
    np.testing.assert_allclose(result.propagation_speedup[:, 0], 1)
    medians = np.median(result.preparation_samples_s, axis=2)
    np.testing.assert_allclose(result.preparation_speedup, medians[:, :1] / medians)
    np.testing.assert_allclose(
        result.preparation_efficiency, result.preparation_speedup / result.parallelism
    )
    assert set(map(tuple, result.execution_order)) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    environment = json.loads(result.environment_json)
    thread_workers = environment["clusters"]["threads_2"]
    process_workers = environment["clusters"]["processes_2"]
    assert len(thread_workers["workers"]) == 1
    assert next(iter(thread_workers["workers"].values()))["nthreads"] == 2
    assert len(process_workers["workers"]) == 2
    assert {w["nthreads"] for w in process_workers["workers"].values()} == {1}
    assert len({w["pid"] for w in process_workers["worker_environment"].values()}) == 2
    for cluster in environment["clusters"].values():
        for worker in cluster["worker_environment"].values():
            assert worker["threadpools"]
            assert all(
                p["num_threads"] == result.blas_threads for p in worker["threadpools"]
            )


def test_cli_artifacts_and_plot_only(scaling_result, tmp_path, monkeypatch):
    import scripts.run_dask_scaling_benchmark as cli

    def compute(config, settings, **kwargs):
        assert settings.qubit_dim == 2
        return scaling_result

    monkeypatch.setattr(cli, "run_dask_scaling_benchmark", compute)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(cli.__file__), "--output-root", str(tmp_path), "--qubit-dim", "2"],
    )
    cli.main()
    path = next(tmp_path.glob("*/dask_scaling_results.h5"))
    loaded = load_result_hdf5(
        path, DaskScalingResult, expected_benchmark_name="dask_scaling"
    )
    np.testing.assert_array_equal(
        loaded.preparation_samples_s, scaling_result.preparation_samples_s
    )
    np.testing.assert_array_equal(loaded.modes, scaling_result.modes)
    metadata_path = path.parent / "benchmark_run.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["benchmark_name"] == "dask_scaling"
    assert "--qubit-dim 2" in metadata["invocation"]
    snapshot = path.parent / "scaling_source.snapshot.py"
    assert snapshot.read_bytes() == (_ROOT / "comparison/dask_scaling.py").read_bytes()
    assert (
        metadata["inputs"]["scaling_source"]["sha256"]
        == hashlib.sha256(snapshot.read_bytes()).hexdigest()
    )
    for name in (
        "git_head.txt",
        "git_scope_snapshot.txt",
        "effective_settings.json",
        "execution_environment.json",
        "dask_scaling_samples.csv",
    ):
        assert (path.parent / name).stat().st_size > 0
    assert (
        len((path.parent / "dask_scaling_samples.csv").read_text().splitlines()) == 11
    )
    assert list(path.parent.glob("*.pdf"))
    original_metadata = metadata_path.read_bytes()
    monkeypatch.setattr(
        cli,
        "run_dask_scaling_benchmark",
        lambda *args, **kwargs: pytest.fail("plot-only must not compute"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(cli.__file__),
            "--plot-only",
            "--results",
            str(path),
            "--settings",
            str(tmp_path / "missing.json"),
        ],
    )
    cli.main()
    assert metadata_path.read_bytes() == original_metadata
    with h5py.File(path) as h5:
        assert len(h5["meta/cli_reports"]) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"parallelism": ()},
        {"parallelism": (2, 4)},
        {"parallelism": (1, 1)},
        {"parallelism": (1, 2.5)},
        {"repeats": 0},
        {"warmups": 0},
        {"blas_threads": 0},
        {"qubit_dim": 1},
        {"hold_time_ns": -1},
        {"dt_ns": float("nan")},
    ],
)
def test_invalid_scaling_settings(kwargs):
    with pytest.raises(ValueError):
        DaskScalingSettings(**kwargs)
