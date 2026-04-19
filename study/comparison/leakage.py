"""Leakage benchmark header.

This module intentionally contains only the benchmark entry-point signature.
Implementation will be added after static-benchmark refactoring is stabilized.
"""

from __future__ import annotations

from study.config import StudyConfig



def run_leakage_benchmark(config: StudyConfig) -> None:
    """Run leakage benchmark (header only, not implemented)."""
    raise NotImplementedError("Leakage benchmark is not implemented yet")
