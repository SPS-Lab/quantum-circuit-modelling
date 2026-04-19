"""CZ benchmark header.

This module intentionally contains only the benchmark entry-point signature.
Implementation will be added after static-benchmark refactoring is stabilized.
"""

from __future__ import annotations

from study.config import StudyConfig



def run_cz_benchmark(config: StudyConfig) -> None:
    """Run CZ-relevant dynamics benchmark (header only, not implemented)."""
    raise NotImplementedError("CZ benchmark is not implemented yet")
