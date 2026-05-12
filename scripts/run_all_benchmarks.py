"""Run all benchmark scripts in sequence."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from runtime_utils import log_progress, run_main_with_timing
_BENCHMARK_SCRIPTS = (
    "run_static_benchmark.py",
    "run_circuit_truncation_benchmark.py",
    "run_duffing_truncation_benchmark.py",
    "run_cz_benchmark.py",
    "run_rx_benchmark.py",
    "run_leakage_flow_benchmark.py",
    "run_runtime_benchmark.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip computations and regenerate all benchmark plots from saved HDF5 files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    child_env = dict(os.environ)
    child_env["PYTHONUNBUFFERED"] = "1"
    for script_name in _BENCHMARK_SCRIPTS:
        script_path = _SCRIPTS_DIR / script_name
        log_progress(f"\n=== Running {script_name} ===")
        cmd = [sys.executable, str(script_path)]
        if args.plot_only:
            cmd.append("--plot-only")
        subprocess.run(cmd, check=True, env=child_env)

    log_progress("\nAll benchmark scripts completed successfully.")


if __name__ == "__main__":
    run_main_with_timing(main)
