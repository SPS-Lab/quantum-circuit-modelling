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
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OMP_DYNAMIC",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "GOTO_NUM_THREADS",
)


@dataclass(frozen=True)
class BenchmarkRunPaths:
    benchmark_name: str
    run_dir: Path
    results_path: Path
    figure_paths: dict[str, Path]
    metadata_path: Path
    git_head_path: Path
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
        git_info = get_git_info(repo_root)
        run_dir = _create_run_dir(
            output_root=resolved_output_root,
            benchmark_name=benchmark_name,
            experiment_name=experiment_name,
            git_info=git_info,
        )
        results_path = run_dir / f"{_sanitize(benchmark_name)}_results.h5"
    if results_path_arg is not None or plot_only:
        git_info = get_git_info(repo_root)

    run_dir = results_path.parent
    resolved_figures = {
        name: run_dir / Path(default_path).name
        for name, default_path in figure_paths.items()
    }
    metadata_path = run_dir / "benchmark_run.json"
    git_head_path = run_dir / "git_head.txt"
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
        git_head_path=git_head_path,
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
            git_info=git_info,
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


def _create_run_dir(
    *,
    output_root: Path,
    benchmark_name: str,
    experiment_name: str | None,
    git_info: dict[str, Any],
) -> Path:
    created_at = datetime.now(timezone.utc)
    stamp = created_at.strftime("%Y%m%d_%H%M%S")
    label = _sanitize(experiment_name or benchmark_name)
    commit_short = _sanitize(str(git_info.get("commit_short", "") or "nogit"))
    dirty_suffix = "_dirty" if bool(git_info.get("dirty")) else ""
    run_dir = output_root / f"{stamp}_{label}_{commit_short}{dirty_suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _materialize_run_metadata(
    *,
    repo_root: Path,
    paths: BenchmarkRunPaths,
    experiment_name: str | None,
    argv: list[str],
    input_files: dict[str, Path],
    git_info: dict[str, Any],
) -> None:
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    _write_git_head_summary(outfile=paths.git_head_path, git_info=git_info)
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
        "hardware": _collect_hardware_snapshot(),
        "runtime_environment": _collect_runtime_environment_snapshot(),
        "git": git_info,
        "git_head": {
            "path": _display_path(paths.git_head_path, repo_root),
        },
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


def get_git_info(repo_root: Path) -> dict[str, Any]:
    return {
        "commit": _run_git(repo_root, "rev-parse", "HEAD"),
        "commit_short": _run_git(repo_root, "rev-parse", "--short", "HEAD"),
        "branch": _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_run_git(repo_root, "status", "--porcelain", "--untracked-files=no")),
    }


def _collect_hardware_snapshot() -> dict[str, Any]:
    total_memory_bytes = _detect_total_memory_bytes()
    return {
        "architecture": platform.machine() or "",
        "cpu_model": _detect_cpu_model(),
        "logical_cpu_count": int(os.cpu_count() or 0),
        "affinity_cpu_count": _detect_affinity_cpu_count(),
        "physical_core_count": _detect_physical_core_count(),
        "total_memory_bytes": int(total_memory_bytes) if total_memory_bytes is not None else None,
        "total_memory_gib": (
            round(float(total_memory_bytes) / float(1 << 30), 3)
            if total_memory_bytes is not None
            else None
        ),
    }


def _collect_runtime_environment_snapshot() -> dict[str, Any]:
    return {
        "pid": int(os.getpid()),
        "threading_env": {name: os.environ.get(name) for name in _THREAD_ENV_VARS},
        "threadpoolctl": _detect_threadpoolctl_snapshot(),
    }


def _detect_threadpoolctl_snapshot() -> dict[str, Any]:
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return {
            "available": False,
            "libraries": [],
        }

    try:
        libraries = [
            {str(key): _json_safe(value) for key, value in entry.items()}
            for entry in threadpool_info()
        ]
    except Exception as exc:  # pragma: no cover - defensive snapshotting only
        return {
            "available": True,
            "libraries": [],
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "available": True,
        "libraries": libraries,
    }


def _detect_cpu_model() -> str:
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        try:
            for line in cpuinfo_path.read_text(encoding="utf-8").splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                if key.strip().lower() == "model name":
                    return value.strip()
        except OSError:
            pass

    processor = platform.processor().strip()
    if processor:
        return processor
    uname_processor = getattr(platform.uname(), "processor", "").strip()
    return uname_processor


def _detect_total_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        if isinstance(page_size, int) and isinstance(phys_pages, int) and page_size > 0 and phys_pages > 0:
            return int(page_size * phys_pages)
    except (AttributeError, OSError, ValueError):
        pass

    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        try:
            for line in meminfo_path.read_text(encoding="utf-8").splitlines():
                if not line.startswith("MemTotal:"):
                    continue
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return int(parts[1]) * 1024
        except OSError:
            pass

    return None


def _detect_affinity_cpu_count() -> int | None:
    if not hasattr(os, "sched_getaffinity"):
        return None
    try:
        return int(len(os.sched_getaffinity(0)))
    except OSError:
        return None


def _detect_physical_core_count() -> int | None:
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        try:
            physical_cores: set[tuple[str, str]] = set()
            current_physical_id = ""
            current_core_id = ""
            found_core_id = False
            for raw_line in cpuinfo_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    if found_core_id:
                        physical_cores.add((current_physical_id, current_core_id))
                    current_physical_id = ""
                    current_core_id = ""
                    found_core_id = False
                    continue
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                normalized = key.strip().lower()
                if normalized == "physical id":
                    current_physical_id = value.strip()
                elif normalized == "core id":
                    current_core_id = value.strip()
                    found_core_id = True

            if found_core_id:
                physical_cores.add((current_physical_id, current_core_id))
            if physical_cores:
                return len(physical_cores)
        except OSError:
            pass

    return None


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


def _write_git_head_summary(*, outfile: Path, git_info: dict[str, Any]) -> None:
    lines = [
        f"commit: {str(git_info.get('commit', ''))}",
        f"commit_short: {str(git_info.get('commit_short', ''))}",
        f"branch: {str(git_info.get('branch', ''))}",
        f"dirty: {bool(git_info.get('dirty'))}",
        "",
    ]
    outfile.write_text("\n".join(lines), encoding="utf-8")


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


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)
