from __future__ import annotations

import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark_run_artifacts import prepare_benchmark_run


def test_prepare_benchmark_run_embeds_commit_in_run_dir_and_summary(tmp_path: Path) -> None:
    repo_root = _ROOT
    figure_path = repo_root / "results" / "dummy_figure.pdf"
    paths = prepare_benchmark_run(
        repo_root=repo_root,
        benchmark_name="static",
        figure_paths={"figure": figure_path},
        results_path_arg=None,
        plot_only=False,
        experiment_name="commit_visibility",
        output_root=tmp_path,
        argv=["python", "scripts/run_static_benchmark.py"],
        input_files={},
    )

    assert paths.run_dir.exists()
    assert paths.metadata_path.exists()
    assert paths.git_head_path.exists()

    git_head_text = paths.git_head_path.read_text(encoding="utf-8")
    assert "commit:" in git_head_text
    assert "commit_short:" in git_head_text

    commit_short = ""
    for line in git_head_text.splitlines():
        if line.startswith("commit_short:"):
            commit_short = line.split(":", 1)[1].strip()
            break

    assert commit_short
    assert commit_short in paths.run_dir.name

    metadata = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
    assert metadata["python_executable"]
    assert metadata["invocation"] == "python scripts/run_static_benchmark.py"

    hardware = metadata["hardware"]
    assert "cpu_model" in hardware
    assert "logical_cpu_count" in hardware
    assert "total_memory_bytes" in hardware

    runtime_environment = metadata["runtime_environment"]
    threading_env = runtime_environment["threading_env"]
    assert "OMP_NUM_THREADS" in threading_env
    assert "OPENBLAS_NUM_THREADS" in threading_env

    threadpoolctl = runtime_environment["threadpoolctl"]
    assert "available" in threadpoolctl
    assert "libraries" in threadpoolctl
