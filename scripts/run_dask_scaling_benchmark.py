"""Benchmark Dask strong scaling for a fixed circuit CZ pulse, with provenance."""

from __future__ import annotations

import argparse
from dataclasses import fields
import csv
import json
from pathlib import Path
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark_cli_reporting import CliReporter
from benchmark_results_io import load_result_hdf5, save_result_hdf5
from benchmark_run_artifacts import prepare_benchmark_run
from comparison.dask_scaling import (
    DaskScalingResult,
    DaskScalingSettings,
    run_dask_scaling_benchmark,
)
from plotting.dask_scaling import plot_dask_scaling_benchmark
from study_config import load_study_config


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settings", type=Path, default=_REPO_ROOT / "params/dask_scaling_params.json"
    )
    for name in ("qubit-dim", "repeats", "warmups", "blas-threads", "seed"):
        parser.add_argument(f"--{name}", type=int, default=None)
    parser.add_argument("--parallelism", nargs="+", type=int, default=None)
    for name in ("target-flux", "ramp-time-ns", "hold-time-ns", "dt-ns"):
        parser.add_argument(f"--{name}", type=float, default=None)
    parser.add_argument("--dashboard-address", default="127.0.0.1:8787")
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def _write_samples(result, path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "mode",
                "parallelism",
                "repeat",
                "build_s",
                "preparation_s",
                "state_propagation_s",
                "propagation_s",
                "build_and_propagation_s",
            )
        )
        for repeat in range(result.repeats):
            b = result.serial_build_samples_s[repeat]
            p = result.serial_preparation_samples_s[repeat]
            s = result.serial_state_propagation_samples_s[repeat]
            writer.writerow(("serial", 1, repeat + 1, b, p, s, p + s, b + p + s))
        for mode_index, mode in enumerate(result.modes):
            for parallel_index, parallelism in enumerate(result.parallelism):
                for repeat in range(result.repeats):
                    index = (mode_index, parallel_index, repeat)
                    b, p, s = (
                        samples[index]
                        for samples in (
                            result.build_samples_s,
                            result.preparation_samples_s,
                            result.state_propagation_samples_s,
                        )
                    )
                    writer.writerow(
                        (mode, parallelism, repeat + 1, b, p, s, p + s, b + p + s)
                    )


def main():
    args = _parse_args()
    settings = None
    if not args.plot_only:
        payload = json.loads(args.settings.read_text(encoding="utf-8"))
        payload.update(
            {
                f.name: getattr(args, f.name)
                for f in fields(DaskScalingSettings)
                if getattr(args, f.name) is not None
            }
        )
        settings = DaskScalingSettings(**payload)
    reporter = CliReporter(
        benchmark_name="dask_scaling", script_name=Path(__file__).name
    )
    paths = prepare_benchmark_run(
        repo_root=_REPO_ROOT,
        benchmark_name="dask_scaling",
        figure_paths={"figure": Path("dask_scaling.pdf")},
        results_path_arg=args.results,
        plot_only=args.plot_only,
        experiment_name=args.experiment_name,
        output_root=args.output_root,
        argv=sys.argv,
        input_files={
            "system_params": _REPO_ROOT / "params/system_params.json",
            "benchmark_params": _REPO_ROOT / "params/benchmark_params.json",
            "dask_scaling_params": args.settings.resolve(),
            # Preserve new/untracked source contents as well as the standard git diff.
            "scaling_source": _REPO_ROOT / "comparison/dask_scaling.py",
            "scaling_script": Path(__file__).resolve(),
            "scaling_plot": _REPO_ROOT / "plotting/dask_scaling.py",
            "simulation_source": _REPO_ROOT / "toolkit/simulation.py",
            "solver_source": _REPO_ROOT / "toolkit/solver.py",
        },
    )
    if args.plot_only:
        result = load_result_hdf5(
            paths.results_path,
            DaskScalingResult,
            expected_benchmark_name="dask_scaling",
        )
    else:
        config = load_study_config(
            system_params_path=_REPO_ROOT / "params/system_params.json",
            study_params_path=_REPO_ROOT / "params/benchmark_params.json",
        )
        result = run_dask_scaling_benchmark(
            config, settings, dashboard_address=args.dashboard_address
        )
        save_result_hdf5(result, paths.results_path, benchmark_name="dask_scaling")
        for name, text in (
            ("effective_settings", result.settings_json),
            ("execution_environment", result.environment_json),
        ):
            (paths.run_dir / f"{name}.json").write_text(text + "\n", encoding="utf-8")
    _write_samples(result, paths.run_dir / "dask_scaling_samples.csv")
    plot_dask_scaling_benchmark(result, paths.figure_paths["figure"])
    effective = json.loads(result.settings_json)
    reporter.line(
        f"Fixed workload: qubit_dim={effective['benchmark']['qubit_dim']}, "
        f"coupler_dim={effective['circuit_config']['hilbert_truncation']['c_truncated_dim']}, "
        f"{len(result.counts)} segments, {len(result.times_ns) - 1} time steps"
    )
    reporter.line(
        f"Repeats={result.repeats}, warmups={result.warmups}, BLAS threads={result.blas_threads}"
    )
    reporter.line(
        "Medians with min/max ranges; speedups use the same Dask mode at p=1."
    )
    reporter.line(
        f"Serial preparation median: {np.median(result.serial_preparation_samples_s):.6f} s"
    )
    for m, mode in enumerate(result.modes):
        for i, p in enumerate(result.parallelism):
            samples = result.preparation_samples_s[m, i]
            reporter.line(
                f"{mode}, p={p}: preparation={np.median(samples):.6f} s "
                f"[{samples.min():.6f}, {samples.max():.6f}], "
                f"speedup={result.preparation_speedup[m, i]:.3f}, "
                f"efficiency={result.preparation_efficiency[m, i]:.3f}; "
                f"preparation+propagation={np.median(result.propagation_samples_s[m, i]):.6f} s, "
                f"combined speedup={result.propagation_speedup[m, i]:.3f}"
            )
    reporter.line(
        f"Maximum amplitude difference: {result.max_amplitude_difference.max():.3e}"
    )
    reporter.line(f"Results and provenance: {paths.run_dir}")
    reporter.add_runtime_line()
    reporter.persist(paths.results_path)


if __name__ == "__main__":
    main()
