"""Shared CLI reporting helpers for benchmark scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import time

from benchmark_results_io import save_cli_report_hdf5
from runtime_utils import format_elapsed_compact, log_progress
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
        log_progress(line_text)
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
        "  Circuit charge-basis cutoffs:",
        f"    q0 transmon charge cutoff: ncut={int(sb.circuit_model.transmon_charge_basis.q0_ncut)}",
        f"    q1 transmon charge cutoff: ncut={int(sb.circuit_model.transmon_charge_basis.q1_ncut)}",
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
            f"q0_truncated_dim={int(sb.circuit_model.hilbert_truncation.q0_truncated_dim)}, "
            f"q1_truncated_dim={int(sb.circuit_model.hilbert_truncation.q1_truncated_dim)}, "
            f"c_truncated_dim={int(sb.circuit_model.hilbert_truncation.c_truncated_dim)}"
        ),
        (
            "  Dressed-state selection truncation: "
            f"n_candidate_states={int(sb.dressed_subspace.n_candidate_states)}, "
            f"selection_mode={str(sb.dressed_subspace.selection_mode)}"
        ),
    ]
    if calibration_mode in ("fixed", "per-flux", "fitted-static", "symbolic-fitted-static"):
        lines.insert(
            6,
            (
                "    transmon_spectral_extraction: "
                f"ncut={int(sb.duffing_model.transmon_spectral_extraction.ncut)}, "
                f"truncated_dim={int(sb.duffing_model.transmon_spectral_extraction.truncated_dim)}"
            ),
        )
    return lines


def build_circuit_truncation_benchmark_extra_lines(config: StudyConfig) -> list[str]:
    """Extra settings specific to circuit truncation benchmark."""
    tb = config.circuit_truncation_benchmark
    flux_values = ", ".join(f"{float(v):.6f}" for v in tb.flux_values)
    circuit_ncut_values = ", ".join(str(int(v)) for v in tb.circuit_ncut_values)
    circuit_qubit_dims = ", ".join(str(int(v)) for v in tb.circuit_qubit_truncated_dim_values)
    circuit_coupler_dims = ", ".join(str(int(v)) for v in tb.circuit_coupler_truncated_dim_values)
    lines = [
        "Circuit truncation benchmark sweep settings:",
        f"  flux_values=[{flux_values}]",
        f"  circuit_ncut_values=[{circuit_ncut_values}]",
        f"  circuit_qubit_truncated_dim_values=[{circuit_qubit_dims}]",
        f"  circuit_coupler_truncated_dim_values=[{circuit_coupler_dims}]",
        f"  circuit_reference_ncut={int(tb.circuit_reference_ncut)}",
        (
            "  circuit_reference_truncation="
            f"{int(tb.circuit_reference_qubit_truncated_dim)}/{int(tb.circuit_reference_coupler_truncated_dim)}"
        ),
        f"  lowest_excited_levels_to_plot={int(tb.lowest_excited_levels_to_plot)}",
    ]
    return lines


def build_duffing_truncation_benchmark_extra_lines(config: StudyConfig) -> list[str]:
    """Extra settings specific to Duffing truncation benchmark."""
    tb = config.duffing_truncation_benchmark
    flux_values = ", ".join(f"{float(v):.6f}" for v in tb.flux_values)
    duffing_ncut_values = ", ".join(str(int(v)) for v in tb.duffing_ncut_values)
    duffing_hilbert_qubit_dims = ", ".join(str(int(v)) for v in tb.duffing_hilbert_qubit_dim_values)
    duffing_hilbert_coupler_dims = ", ".join(str(int(v)) for v in tb.duffing_hilbert_coupler_dim_values)
    mode = str(tb.duffing_calibration_mode)
    lines = [
        "Duffing truncation benchmark sweep settings:",
        f"  flux_values=[{flux_values}]",
        f"  duffing_ncut_values=[{duffing_ncut_values}]",
        f"  duffing_truncated_dim={int(tb.duffing_truncated_dim)}",
        f"  duffing_hilbert_qubit_dim_values=[{duffing_hilbert_qubit_dims}]",
        f"  duffing_hilbert_coupler_dim_values=[{duffing_hilbert_coupler_dims}]",
        (
            "  duffing_reference_hilbert_truncation="
            f"{int(tb.duffing_reference_hilbert_qubit_dim)}/{int(tb.duffing_reference_hilbert_coupler_dim)}"
        ),
        f"  duffing_calibration_mode={mode}",
        f"  circuit_reference_ncut={int(tb.circuit_reference_ncut)}",
        (
            "  circuit_reference_truncation="
            f"{int(tb.circuit_reference_qubit_truncated_dim)}/{int(tb.circuit_reference_coupler_truncated_dim)}"
        ),
        f"  lowest_excited_levels_to_plot={int(tb.lowest_excited_levels_to_plot)}",
    ]
    return lines
