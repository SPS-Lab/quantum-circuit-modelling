"""Shared helpers for compact runtime reporting in script entrypoints."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
import threading
import time


def format_elapsed_compact(seconds: float) -> str:
    """Format elapsed wall-clock time in a short human-readable form."""
    total_ms = int(round(max(0.0, float(seconds)) * 1000.0))
    if total_ms < 1000:
        return f"{total_ms}ms"

    total_s, _ = divmod(total_ms, 1000)
    if total_s < 60:
        return f"{(total_ms / 1000.0):.1f}s"

    total_m, sec = divmod(total_s, 60)
    if total_m < 60:
        return f"{total_m}m{sec:02d}s"

    hours, minutes = divmod(total_m, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"


def log_progress(text: str) -> None:
    """Print a progress line immediately so long runs stay visible in terminals."""
    print(str(text), flush=True)


@contextmanager
def progress_heartbeat(label: str, *, interval_s: float = 30.0):
    """Emit start/finish lines and periodic keep-alive updates for long sections."""
    interval = max(float(interval_s), 1.0)
    started = time.perf_counter()
    stop_event = threading.Event()

    def _worker() -> None:
        while not stop_event.wait(interval):
            elapsed = time.perf_counter() - started
            log_progress(f"{label} still running after {format_elapsed_compact(elapsed)}")

    worker = threading.Thread(target=_worker, name="progress-heartbeat", daemon=True)
    log_progress(f"{label} started")
    worker.start()
    try:
        yield
    finally:
        stop_event.set()
        worker.join(timeout=0.1)
        elapsed = time.perf_counter() - started
        log_progress(f"{label} finished in {format_elapsed_compact(elapsed)}")


def run_main_with_timing(main_fn: Callable[[], None], *, label: str = "Total runtime") -> None:
    """Run a script main function and print total elapsed time on exit."""
    started = time.perf_counter()
    try:
        main_fn()
    finally:
        elapsed = time.perf_counter() - started
        log_progress(f"{label}: {format_elapsed_compact(elapsed)}")
