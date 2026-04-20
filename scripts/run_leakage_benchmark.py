"""Header script for leakage benchmark (not implemented yet)."""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from comparison.leakage import run_leakage_benchmark
from study_config import load_study_config



def main() -> None:
    repo_root = _REPO_ROOT
    config = load_study_config(
        repo_root / "params" / "system_params.json",
        repo_root / "params" / "static_benchmark_params.json",
    )
    run_leakage_benchmark(config)


if __name__ == "__main__":
    main()
