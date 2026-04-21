"""Run all benchmark scripts in sequence."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
_BENCHMARK_SCRIPTS = (
    "run_static_benchmark.py",
    "run_leakage_benchmark.py",
    "run_cz_benchmark.py",
    "run_truncation_benchmark.py",
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
    for script_name in _BENCHMARK_SCRIPTS:
        script_path = _SCRIPTS_DIR / script_name
        print(f"\n=== Running {script_name} ===")
        cmd = [sys.executable, str(script_path)]
        if args.plot_only:
            cmd.append("--plot-only")
        subprocess.run(cmd, check=True)

    print("\nAll benchmark scripts completed successfully.")


if __name__ == "__main__":
    main()
