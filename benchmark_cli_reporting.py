"""Shared CLI reporting helpers for benchmark scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import time

from benchmark_results_io import save_cli_report_hdf5
from runtime_utils import format_elapsed_compact
from study_config import StudyConfig


@dataclass
class CliReporter:
    """Collect lines printed to CLI and persist them into benchmark HDF5 files."""

    benchmark_name: str
    script_name: str
    started_utc_iso: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    _started_perf: float = field(default_factory=time.perf_counter)
    lines: list[str] = field(default_factory=list)

    def line(self, text: str = "") -> None:
        line_text = str(text)
        print(line_text)
        self.lines.append(line_text)

    def add_runtime_line(self, *, label: str = "Total runtime") -> None:
        elapsed = time.perf_counter() - self._started_perf
        self.line(f"{label}: {format_elapsed_compact(elapsed)}")

    def persist(self, results_path: Path) -> None:
        finished_utc_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_cli_report_hdf5(
            results_path,
            benchmark_name=self.benchmark_name,
            script_name=self.script_name,
            lines=self.lines,
            started_utc_iso=self.started_utc_iso,
            finished_utc_iso=finished_utc_iso,
        )


def build_common_truncation_lines(config: StudyConfig) -> list[str]:
    """Human-readable truncation settings used across benchmarks."""
    sb = config.static_benchmark
    calibration_mode = str(sb.duffing_model.calibration_mode)
    lines = [
        "Truncation settings:",
        "  System basis cutoffs:",
        f"    q1 transmon charge cutoff: ncut={int(config.system.q1.ncut)}",
        f"    q2 transmon charge cutoff: ncut={int(config.system.q2.ncut)}",
        "  Duffing calibration extraction:",
        f"    calibration_mode={calibration_mode}",
        "  Dressed-subspace model truncation:",
        (
            "    duffing hilbert_truncation: "
            f"nlevels_qubit={int(sb.duffing_model.hilbert_truncation.nlevels_qubit)}, "
            f"nlevels_coupler={int(sb.duffing_model.hilbert_truncation.nlevels_coupler)}"
        ),
        (
            "    circuit hilbert_truncation: "
            f"q1_truncated_dim={int(sb.circuit_model.hilbert_truncation.q1_truncated_dim)}, "
            f"q2_truncated_dim={int(sb.circuit_model.hilbert_truncation.q2_truncated_dim)}, "
            f"c_truncated_dim={int(sb.circuit_model.hilbert_truncation.c_truncated_dim)}"
        ),
        (
            "  Dressed-state selection truncation: "
            f"n_candidate_states={int(sb.dressed_subspace.n_candidate_states)}, "
            f"selection_mode={str(sb.dressed_subspace.selection_mode)}"
        ),
    ]
    if calibration_mode in ("fixed", "per-flux"):
        lines.insert(
            6,
            (
                "    transmon_spectral_extraction: "
                f"ncut={int(sb.duffing_model.transmon_spectral_extraction.ncut)}, "
                f"truncated_dim={int(sb.duffing_model.transmon_spectral_extraction.truncated_dim)}"
            ),
        )
    return lines


def build_truncation_benchmark_extra_lines(config: StudyConfig) -> list[str]:
    """Extra truncation-sweep settings specific to truncation benchmark."""
    tb = config.truncation_benchmark
    ncut_values = ", ".join(str(int(v)) for v in tb.duffing_ncut_values)
    mode = str(tb.duffing_calibration_mode)
    lines = [
        "Truncation benchmark sweep settings:",
        f"  fixed_flux={float(tb.fixed_flux):.6f}",
        f"  duffing_calibration_mode={mode}",
        f"  circuit_reference_ncut={int(tb.circuit_reference_ncut)}",
        f"  lowest_excited_levels_to_plot={int(tb.lowest_excited_levels_to_plot)}",
    ]
    if mode in ("fixed", "per-flux"):
        lines.insert(2, f"  duffing_ncut_values=[{ncut_values}]")
        lines.insert(3, f"  duffing_truncated_dim={int(tb.duffing_truncated_dim)}")
    return lines
