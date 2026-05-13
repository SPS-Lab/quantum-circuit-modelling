"""Helpers for benchmark run directories and lightweight provenance snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
from typing import Any


_SCHEMA_VERSION = "benchmark_run_v1"
_DEFAULT_OUTPUT_ROOT = Path("results/experiments")


@dataclass(frozen=True)
class BenchmarkRunPaths:
    benchmark_name: str
    run_dir: Path
    results_path: Path
    figure_paths: dict[str, Path]
    metadata_path: Path
    git_snapshot_path: Path
    input_snapshot_paths: dict[str, Path]


def prepare_benchmark_run(
    *,
    repo_root: Path,
    benchmark_name: str,
    figure_paths: dict[str, Path],
    results_path_arg: Path | None,
    plot_only: bool,
    experiment_name: str | None = None,
    output_root: Path | None = None,
    argv: list[str] | None = None,
    input_files: dict[str, Path] | None = None,
) -> BenchmarkRunPaths:
    """Resolve run-local artifact paths and optionally create a new run directory."""
    resolved_output_root = _resolve_output_root(repo_root, output_root)
    if results_path_arg is not None:
        results_path = _resolve_repo_relative(repo_root, results_path_arg)
    elif plot_only:
        results_path = _resolve_plot_only_results_path(
            repo_root=repo_root,
            benchmark_name=benchmark_name,
            figure_paths=figure_paths,
            output_root=resolved_output_root,
        )
    else:
        run_dir = _create_run_dir(
            output_root=resolved_output_root,
            benchmark_name=benchmark_name,
            experiment_name=experiment_name,
        )
        results_path = run_dir / f"{_sanitize(benchmark_name)}_results.h5"

    run_dir = results_path.parent
    resolved_figures = {
        name: run_dir / Path(default_path).name
        for name, default_path in figure_paths.items()
    }
    metadata_path = run_dir / "benchmark_run.json"
    git_snapshot_path = run_dir / "git_scope_snapshot.txt"
    snapshot_paths = {
        name: run_dir / f"{_sanitize(name)}.snapshot{Path(path).suffix or '.txt'}"
        for name, path in (input_files or {}).items()
    }

    paths = BenchmarkRunPaths(
        benchmark_name=str(benchmark_name),
        run_dir=run_dir,
        results_path=results_path,
        figure_paths=resolved_figures,
        metadata_path=metadata_path,
        git_snapshot_path=git_snapshot_path,
        input_snapshot_paths=snapshot_paths,
    )
    if not plot_only:
        _materialize_run_metadata(
            repo_root=repo_root,
            paths=paths,
            experiment_name=experiment_name,
            argv=argv or sys.argv,
            input_files=input_files or {},
        )
    return paths


def _resolve_plot_only_results_path(
    *,
    repo_root: Path,
    benchmark_name: str,
    figure_paths: dict[str, Path],
    output_root: Path,
) -> Path:
    latest = find_latest_results_path(output_root=output_root, benchmark_name=benchmark_name)
    if latest is not None:
        return latest
    first_figure = next(iter(figure_paths.values()))
    return first_figure.with_suffix(".h5")


def find_latest_results_path(*, output_root: Path, benchmark_name: str) -> Path | None:
    pattern = f"*/{_sanitize(benchmark_name)}_results.h5"
    candidates = [path for path in output_root.glob(pattern) if path.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda path: (path.parent.name, path.stat().st_mtime_ns))
    return candidates[-1]


def _resolve_output_root(repo_root: Path, output_root: Path | None) -> Path:
    root = _DEFAULT_OUTPUT_ROOT if output_root is None else output_root
    return _resolve_repo_relative(repo_root, root)


def _resolve_repo_relative(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path)


def _create_run_dir(*, output_root: Path, benchmark_name: str, experiment_name: str | None) -> Path:
    created_at = datetime.now(timezone.utc)
    stamp = created_at.strftime("%Y%m%d_%H%M%S")
    label = _sanitize(experiment_name or benchmark_name)
    run_dir = output_root / f"{stamp}_{label}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _materialize_run_metadata(
    *,
    repo_root: Path,
    paths: BenchmarkRunPaths,
    experiment_name: str | None,
    argv: list[str],
    input_files: dict[str, Path],
) -> None:
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    git_info = _get_git_info(repo_root)
    git_snapshot = _write_git_snapshot(repo_root=repo_root, outfile=paths.git_snapshot_path)
    inputs_payload: dict[str, Any] = {}
    for name, source_path in input_files.items():
        src = _resolve_repo_relative(repo_root, Path(source_path))
        dst = paths.input_snapshot_paths[name]
        if src.exists():
            shutil.copy2(src, dst)
        inputs_payload[name] = {
            "source_path": _display_path(src, repo_root),
            "source_exists": bool(src.exists()),
            "snapshot_path": _display_path(dst, repo_root),
            "sha256": _sha256_file(src) if src.exists() else "",
            "size_bytes": int(src.stat().st_size) if src.exists() else 0,
        }

    payload = {
        "schema_version": _SCHEMA_VERSION,
        "benchmark_name": paths.benchmark_name,
        "experiment_name": str(experiment_name or paths.benchmark_name),
        "created_utc_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_root": str(repo_root),
        "cwd": str(Path.cwd()),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "user": os.environ.get("USER", ""),
        "invocation": shlex.join([str(token) for token in argv]),
        "git": git_info,
        "git_scope_snapshot": git_snapshot,
        "inputs": inputs_payload,
        "artifacts": {
            "results_path": _display_path(paths.results_path, repo_root),
            "figures": {
                name: _display_path(path, repo_root) for name, path in paths.figure_paths.items()
            },
        },
    }
    paths.metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _get_git_info(repo_root: Path) -> dict[str, Any]:
    return {
        "commit": _run_git(repo_root, "rev-parse", "HEAD"),
        "commit_short": _run_git(repo_root, "rev-parse", "--short", "HEAD"),
        "branch": _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_run_git(repo_root, "status", "--porcelain", "--untracked-files=no")),
    }


def _write_git_snapshot(*, repo_root: Path, outfile: Path) -> dict[str, Any]:
    staged = _run_git_text(repo_root, "diff", "--cached", "--", ".")
    unstaged = _run_git_text(repo_root, "diff", "--", ".")
    untracked = _run_git_text(repo_root, "ls-files", "--others", "--exclude-standard", "--", ".")
    untracked_files = [line.strip() for line in untracked.splitlines() if line.strip()]

    lines = [
        "# Git snapshot for benchmark reproducibility",
        "# scope: .",
        "",
        "## staged_diff",
        staged.rstrip() if staged.strip() else "# (no staged changes)",
        "",
        "## unstaged_diff",
        unstaged.rstrip() if unstaged.strip() else "# (no unstaged changes)",
        "",
        "## untracked_files",
        *(untracked_files if untracked_files else ["# (no untracked files)"]),
        "",
    ]
    payload = "\n".join(lines)
    outfile.write_text(payload, encoding="utf-8")
    return {
        "scope_paths": ["."],
        "path": _display_path(outfile, repo_root),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "dirty": bool(staged.strip() or unstaged.strip() or untracked_files),
    }


def _run_git(repo_root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _run_git_text(repo_root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else ""


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sanitize(value: str) -> str:
    keep = [ch.lower() if ch.isalnum() else "_" for ch in str(value).strip()]
    compact = "".join(keep).strip("_")
    while "__" in compact:
        compact = compact.replace("__", "_")
    return compact or "run"
