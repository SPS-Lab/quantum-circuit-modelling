"""Shared helpers for compact runtime reporting in script entrypoints."""

from __future__ import annotations

from collections.abc import Callable
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


def run_main_with_timing(main_fn: Callable[[], None], *, label: str = "Total runtime") -> None:
    """Run a script main function and print total elapsed time on exit."""
    started = time.perf_counter()
    try:
        main_fn()
    finally:
        elapsed = time.perf_counter() - started
        print(f"{label}: {format_elapsed_compact(elapsed)}")
