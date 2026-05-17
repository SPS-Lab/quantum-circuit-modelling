"""Artifacts for reusing static fitted lower-model parameters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_results_io import load_result_hdf5


_SCHEMA_VERSION = "static_fitted_models_v1"


@dataclass(frozen=True)
class StaticFittedModelsArtifact:
    flux_values: np.ndarray
    effective_parameters: dict[str, np.ndarray]
    effective_fit_coefficient_names: dict[str, np.ndarray]
    effective_fit_coefficients: dict[str, np.ndarray]
    duffing_mode_parameters: dict[str, np.ndarray]
    duffing_symbolic_coefficient_names: dict[str, np.ndarray]
    duffing_symbolic_coefficients: dict[str, np.ndarray]
    circuit_parameters: dict[str, np.ndarray]
    summary: dict[str, float]
    sweep_target: str
    duffing_calibration_mode: str
    effective_derivation_source: str
    effective_fit_basis: str


def build_static_fitted_models_artifact(result: Any, *, config: Any) -> StaticFittedModelsArtifact:
    return StaticFittedModelsArtifact(
        flux_values=np.asarray(result.flux_values, dtype=float),
        effective_parameters=_copy_float_mapping(result.effective_parameters),
        effective_fit_coefficient_names=_copy_string_mapping(result.effective_fit_coefficient_names),
        effective_fit_coefficients=_copy_float_mapping(result.effective_fit_coefficients),
        duffing_mode_parameters=_copy_float_mapping(result.duffing_mode_parameters),
        duffing_symbolic_coefficient_names=_copy_string_mapping(result.duffing_symbolic_coefficient_names),
        duffing_symbolic_coefficients=_copy_float_mapping(result.duffing_symbolic_coefficients),
        circuit_parameters=_copy_float_mapping(result.circuit_parameters),
        summary={str(key): float(value) for key, value in result.summary.items()},
        sweep_target=str(config.static_benchmark.flux_control.sweep_target),
        duffing_calibration_mode=str(config.static_benchmark.duffing_model.calibration_mode),
        effective_derivation_source=str(config.static_benchmark.effective_model.derivation_source),
        effective_fit_basis=str(config.static_benchmark.effective_model.fit_basis),
    )


def save_static_fitted_models_artifact(artifact: StaticFittedModelsArtifact, outfile: Path) -> None:
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "saved_utc_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sweep_target": artifact.sweep_target,
        "duffing_calibration_mode": artifact.duffing_calibration_mode,
        "effective_derivation_source": artifact.effective_derivation_source,
        "effective_fit_basis": artifact.effective_fit_basis,
        "flux_values": artifact.flux_values.tolist(),
        "effective_parameters": _mapping_to_lists(artifact.effective_parameters),
        "effective_fit_coefficient_names": _mapping_to_lists(artifact.effective_fit_coefficient_names),
        "effective_fit_coefficients": _mapping_to_lists(artifact.effective_fit_coefficients),
        "duffing_mode_parameters": _mapping_to_lists(artifact.duffing_mode_parameters),
        "duffing_symbolic_coefficient_names": _mapping_to_lists(artifact.duffing_symbolic_coefficient_names),
        "duffing_symbolic_coefficients": _mapping_to_lists(artifact.duffing_symbolic_coefficients),
        "circuit_parameters": _mapping_to_lists(artifact.circuit_parameters),
        "summary": {str(key): float(value) for key, value in artifact.summary.items()},
    }
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_static_fitted_models_artifact(path: Path) -> StaticFittedModelsArtifact:
    resolved = resolve_static_fitted_artifact_path(path)
    if resolved.suffix.lower() == ".h5":
        from comparison.static import StaticBenchmarkResult

        static_result = load_result_hdf5(
            resolved,
            StaticBenchmarkResult,
            expected_benchmark_name="static",
        )
        return StaticFittedModelsArtifact(
            flux_values=np.asarray(static_result.flux_values, dtype=float),
            effective_parameters=_copy_float_mapping(static_result.effective_parameters),
            effective_fit_coefficient_names=_copy_string_mapping(static_result.effective_fit_coefficient_names),
            effective_fit_coefficients=_copy_float_mapping(static_result.effective_fit_coefficients),
            duffing_mode_parameters=_copy_float_mapping(static_result.duffing_mode_parameters),
            duffing_symbolic_coefficient_names=_copy_string_mapping(static_result.duffing_symbolic_coefficient_names),
            duffing_symbolic_coefficients=_copy_float_mapping(static_result.duffing_symbolic_coefficients),
            circuit_parameters=_copy_float_mapping(static_result.circuit_parameters),
            summary={str(key): float(value) for key, value in static_result.summary.items()},
            sweep_target="",
            duffing_calibration_mode="",
            effective_derivation_source="",
            effective_fit_basis="",
        )

    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if str(payload.get("schema_version", "")) != _SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported static fitted artifact schema {payload.get('schema_version')!r} in {resolved}"
        )
    return StaticFittedModelsArtifact(
        flux_values=np.asarray(payload["flux_values"], dtype=float),
        effective_parameters=_mapping_from_payload(payload["effective_parameters"], dtype=float),
        effective_fit_coefficient_names=_mapping_from_payload(payload["effective_fit_coefficient_names"], dtype=str),
        effective_fit_coefficients=_mapping_from_payload(payload["effective_fit_coefficients"], dtype=float),
        duffing_mode_parameters=_mapping_from_payload(payload["duffing_mode_parameters"], dtype=float),
        duffing_symbolic_coefficient_names=_mapping_from_payload(payload["duffing_symbolic_coefficient_names"], dtype=str),
        duffing_symbolic_coefficients=_mapping_from_payload(payload["duffing_symbolic_coefficients"], dtype=float),
        circuit_parameters=_mapping_from_payload(payload["circuit_parameters"], dtype=float),
        summary={str(key): float(value) for key, value in payload["summary"].items()},
        sweep_target=str(payload.get("sweep_target", "")),
        duffing_calibration_mode=str(payload.get("duffing_calibration_mode", "")),
        effective_derivation_source=str(payload.get("effective_derivation_source", "")),
        effective_fit_basis=str(payload.get("effective_fit_basis", "")),
    )


def resolve_static_fitted_artifact_path(path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        json_path = candidate / "static_fitted_parameters.json"
        if json_path.exists():
            return json_path
        h5_path = candidate / "static_results.h5"
        if h5_path.exists():
            return h5_path
        raise FileNotFoundError(
            "Expected static fitted artifact directory to contain "
            "'static_fitted_parameters.json' or 'static_results.h5'"
        )
    return candidate


def build_static_fitted_latex_table(
    artifact: StaticFittedModelsArtifact,
    *,
    git_info: dict[str, Any] | None = None,
    experiment_folder_name: str | None = None,
) -> str:
    lines = [
        "% Auto-generated static fitted-parameter appendix tables.",
        "% Paste the tabular blocks where needed in your LaTeX source.",
    ]
    lines.extend(
        _build_table_block(
            title="Effective fitted coefficients",
            label="Effective parameter",
            coefficient_names=artifact.effective_fit_coefficient_names,
            coefficients=artifact.effective_fit_coefficients,
            parameter_order=("w0", "w1", "J", "zeta"),
            provenance_lines=_build_provenance_comment_lines(
                git_info=git_info,
                experiment_folder_name=experiment_folder_name,
            ),
        )
    )
    lines.append("")
    if artifact.duffing_symbolic_coefficients:
        lines.extend(
            _build_table_block(
                title="Symbolic Duffing fitted coefficients",
                label="Duffing parameter",
                coefficient_names=artifact.duffing_symbolic_coefficient_names,
                coefficients=artifact.duffing_symbolic_coefficients,
                parameter_order=("w0", "w1", "alpha0", "alpha1", "g0c", "g1c"),
                provenance_lines=_build_provenance_comment_lines(
                    git_info=git_info,
                    experiment_folder_name=experiment_folder_name,
                ),
            )
        )
    else:
        lines.append("% No symbolic Duffing coefficients were available for this run.")
    return "\n".join(lines).rstrip() + "\n"


def build_static_fitted_markdown_table(
    artifact: StaticFittedModelsArtifact,
    *,
    git_info: dict[str, Any] | None = None,
    experiment_folder_name: str | None = None,
) -> str:
    lines = [
        "<!-- Auto-generated static fitted-parameter appendix tables. -->",
        "<!-- Markdown companion to static_fitted_parameters_table.tex -->",
    ]
    lines.extend(
        _build_markdown_table_block(
            title="Effective fitted coefficients",
            label="Effective parameter",
            coefficient_names=artifact.effective_fit_coefficient_names,
            coefficients=artifact.effective_fit_coefficients,
            parameter_order=("w0", "w1", "J", "zeta"),
            provenance_lines=_build_provenance_comment_lines(
                git_info=git_info,
                experiment_folder_name=experiment_folder_name,
            ),
        )
    )
    lines.append("")
    if artifact.duffing_symbolic_coefficients:
        lines.extend(
            _build_markdown_table_block(
                title="Symbolic Duffing fitted coefficients",
                label="Duffing parameter",
                coefficient_names=artifact.duffing_symbolic_coefficient_names,
                coefficients=artifact.duffing_symbolic_coefficients,
                parameter_order=("w0", "w1", "alpha0", "alpha1", "g0c", "g1c"),
                provenance_lines=_build_provenance_comment_lines(
                    git_info=git_info,
                    experiment_folder_name=experiment_folder_name,
                ),
            )
        )
    else:
        lines.append("<!-- No symbolic Duffing coefficients were available for this run. -->")
    return "\n".join(lines).rstrip() + "\n"


def _build_provenance_comment_lines(
    *,
    git_info: dict[str, Any] | None,
    experiment_folder_name: str | None,
) -> list[str]:
    lines: list[str] = []
    if experiment_folder_name:
        lines.append(f"% Experiment folder: {experiment_folder_name}")
    if git_info is None:
        return lines

    commit = str(git_info.get("commit", "")).strip()
    commit_short = str(git_info.get("commit_short", "")).strip()
    branch = str(git_info.get("branch", "")).strip()
    dirty = bool(git_info.get("dirty", False))
    if commit:
        lines.append(
            "% Git provenance: "
            f"commit={commit}"
            + (f" (short={commit_short})" if commit_short else "")
            + (f", branch={branch}" if branch else "")
            + (", dirty=true" if dirty else ", dirty=false")
        )
    return lines


def _build_table_block(
    *,
    title: str,
    label: str,
    coefficient_names: dict[str, np.ndarray],
    coefficients: dict[str, np.ndarray],
    parameter_order: tuple[str, ...] | None = None,
    provenance_lines: list[str] | None = None,
) -> list[str]:
    lines = [
        f"% {title}",
        *(provenance_lines or []),
        r"\begin{tabular}{lll}",
        r"\hline",
        f"{label} & Coefficient & Value (GHz) " + r"\\",
        r"\hline",
    ]
    ordered_parameter_names = _ordered_parameter_names(coefficients, parameter_order=parameter_order)
    for parameter_name in ordered_parameter_names:
        names = [str(value) for value in np.asarray(coefficient_names[parameter_name], dtype=str).ravel()]
        values = [float(value) for value in np.asarray(coefficients[parameter_name], dtype=float).ravel()]
        if len(names) != len(values):
            raise ValueError(f"Coefficient name/value mismatch for {parameter_name!r}")
        for coeff_name, coeff_value in zip(names, values):
            lines.append(
                f"{_latex_escape(parameter_name)} & {_latex_escape(coeff_name)} & {coeff_value:.6e} " + r"\\"
            )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return lines


def _build_markdown_table_block(
    *,
    title: str,
    label: str,
    coefficient_names: dict[str, np.ndarray],
    coefficients: dict[str, np.ndarray],
    parameter_order: tuple[str, ...] | None = None,
    provenance_lines: list[str] | None = None,
) -> list[str]:
    lines = [f"## {title}"]
    for line in (provenance_lines or []):
        if line.startswith("% "):
            lines.append(f"<!-- {line[2:]} -->")
        elif line.startswith("%"):
            lines.append(f"<!-- {line[1:]} -->")
        else:
            lines.append(f"<!-- {line} -->")
    lines.extend(
        [
            "",
            f"| {label} | Coefficient | Value (GHz) |",
            "| --- | --- | ---: |",
        ]
    )
    ordered_parameter_names = _ordered_parameter_names(coefficients, parameter_order=parameter_order)
    for parameter_name in ordered_parameter_names:
        names = [str(value) for value in np.asarray(coefficient_names[parameter_name], dtype=str).ravel()]
        values = [float(value) for value in np.asarray(coefficients[parameter_name], dtype=float).ravel()]
        if len(names) != len(values):
            raise ValueError(f"Coefficient name/value mismatch for {parameter_name!r}")
        for coeff_name, coeff_value in zip(names, values):
            lines.append(f"| {parameter_name} | {coeff_name} | {coeff_value:.6e} |")
    return lines


def _ordered_parameter_names(
    coefficients: dict[str, np.ndarray],
    *,
    parameter_order: tuple[str, ...] | None,
) -> list[str]:
    if parameter_order is None:
        return sorted(coefficients)
    ordered = [name for name in parameter_order if name in coefficients]
    remaining = sorted(name for name in coefficients if name not in ordered)
    return ordered + remaining


def _mapping_to_lists(mapping: dict[str, np.ndarray]) -> dict[str, list[Any]]:
    return {
        str(key): np.asarray(value).tolist()
        for key, value in mapping.items()
    }


def _mapping_from_payload(payload: dict[str, Any], *, dtype: type[Any]) -> dict[str, np.ndarray]:
    return {
        str(key): np.asarray(value, dtype=dtype)
        for key, value in payload.items()
    }


def _copy_float_mapping(mapping: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        str(key): np.asarray(value, dtype=float)
        for key, value in mapping.items()
    }


def _copy_string_mapping(mapping: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        str(key): np.asarray(value, dtype=str)
        for key, value in mapping.items()
    }


def _latex_escape(value: str) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )
