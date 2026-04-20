"""Run all benchmark scripts in sequence."""

from __future__ import annotations

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


def main() -> None:
    for script_name in _BENCHMARK_SCRIPTS:
        script_path = _SCRIPTS_DIR / script_name
        print(f"\n=== Running {script_name} ===")
        subprocess.run([sys.executable, str(script_path)], check=True)

    print("\nAll benchmark scripts completed successfully.")


if __name__ == "__main__":
    main()
