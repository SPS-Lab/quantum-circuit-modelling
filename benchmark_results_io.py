"""Utilities for persisting benchmark dataclass results to HDF5."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import h5py
import numpy as np

_RESULT_GROUP = "result"
_META_GROUP = "meta"
_SCHEMA_VERSION = "benchmark_result_v1"
_DICT_KIND = "dict"

T = TypeVar("T")


def default_results_path_for_figure(figure_path: Path) -> Path:
    """Return the default results file path for a benchmark figure."""
    return figure_path.with_suffix(".h5")


def save_result_hdf5(
    result: Any,
    outfile: Path,
    *,
    benchmark_name: str,
) -> None:
    """Save a dataclass benchmark result to HDF5."""
    if not is_dataclass(result):
        raise TypeError("save_result_hdf5 expects a dataclass instance")

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(outfile, "w") as h5:
        h5.attrs["schema_version"] = _SCHEMA_VERSION
        h5.attrs["benchmark_name"] = str(benchmark_name)
        h5.attrs["result_class"] = type(result).__name__
        h5.attrs["saved_utc_iso"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        root = h5.create_group(_RESULT_GROUP)
        for f in fields(result):
            _write_value(root, f.name, getattr(result, f.name))


def save_cli_report_hdf5(
    outfile: Path,
    *,
    benchmark_name: str,
    script_name: str,
    lines: list[str],
    started_utc_iso: str,
    finished_utc_iso: str,
) -> None:
    """Append a CLI report for one script run to an HDF5 results file."""
    outfile.parent.mkdir(parents=True, exist_ok=True)
    utf8 = h5py.string_dtype(encoding="utf-8")
    line_array = np.asarray([str(line) for line in lines], dtype=utf8)
    text_blob = "\n".join(str(line) for line in lines)

    with h5py.File(outfile, "a") as h5:
        meta = h5.require_group(_META_GROUP)
        reports = meta.require_group("cli_reports")
        run_index = len(reports)
        run_group = reports.create_group(f"run_{run_index:04d}")
        run_group.attrs["benchmark_name"] = str(benchmark_name)
        run_group.attrs["script_name"] = str(script_name)
        run_group.attrs["started_utc_iso"] = str(started_utc_iso)
        run_group.attrs["finished_utc_iso"] = str(finished_utc_iso)
        run_group.attrs["line_count"] = int(len(lines))
        run_group.create_dataset("lines", data=line_array)
        run_group.create_dataset("text", data=np.array(text_blob, dtype=utf8))

        if "last_cli_report_lines" in meta:
            del meta["last_cli_report_lines"]
        if "last_cli_report_text" in meta:
            del meta["last_cli_report_text"]
        meta.create_dataset("last_cli_report_lines", data=line_array)
        meta.create_dataset("last_cli_report_text", data=np.array(text_blob, dtype=utf8))
        meta.attrs["last_cli_report_benchmark_name"] = str(benchmark_name)
        meta.attrs["last_cli_report_script_name"] = str(script_name)
        meta.attrs["last_cli_report_started_utc_iso"] = str(started_utc_iso)
        meta.attrs["last_cli_report_finished_utc_iso"] = str(finished_utc_iso)
        meta.attrs["last_cli_report_line_count"] = int(len(lines))


def load_result_hdf5(
    infile: Path,
    result_type: type[T],
    *,
    expected_benchmark_name: str,
) -> T:
    """Load a dataclass benchmark result from HDF5."""
    if not is_dataclass(result_type):
        raise TypeError("load_result_hdf5 expects a dataclass type")

    with h5py.File(infile, "r") as h5:
        if "schema_version" not in h5.attrs:
            raise ValueError(f"Missing required file attribute 'schema_version' in {infile}")
        schema_version = _attr_text(h5.attrs["schema_version"])
        if schema_version != _SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {schema_version!r}; expected {_SCHEMA_VERSION!r}"
            )

        if "benchmark_name" not in h5.attrs:
            raise ValueError(f"Missing required file attribute 'benchmark_name' in {infile}")
        benchmark_name = _attr_text(h5.attrs["benchmark_name"])
        if benchmark_name != expected_benchmark_name:
            raise ValueError(
                f"Results file benchmark {benchmark_name!r} does not match "
                f"expected {expected_benchmark_name!r}"
            )
        if "result_class" not in h5.attrs:
            raise ValueError(f"Missing required file attribute 'result_class' in {infile}")
        result_class = _attr_text(h5.attrs["result_class"])
        expected_class = result_type.__name__
        if result_class != expected_class:
            raise ValueError(
                f"Results file class {result_class!r} does not match expected {expected_class!r}"
            )

        if _RESULT_GROUP not in h5:
            raise ValueError(f"Missing {_RESULT_GROUP!r} group in {infile}")
        root = h5[_RESULT_GROUP]

        kwargs: dict[str, Any] = {}
        for f in fields(result_type):
            if f.name not in root:
                raise ValueError(f"Missing field {f.name!r} in results file {infile}")
            kwargs[f.name] = _read_value(root[f.name])

    return result_type(**kwargs)


def _write_value(group: h5py.Group, key: str, value: Any) -> None:
    if isinstance(value, dict):
        sub = group.create_group(str(key))
        sub.attrs["kind"] = _DICT_KIND
        for sub_key, sub_value in value.items():
            _write_value(sub, str(sub_key), sub_value)
        return

    if isinstance(value, str):
        utf8 = h5py.string_dtype(encoding="utf-8")
        group.create_dataset(str(key), data=np.array(value, dtype=utf8))
        return

    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S"}:
            utf8 = h5py.string_dtype(encoding="utf-8")
            group.create_dataset(str(key), data=np.asarray(value, dtype=object), dtype=utf8)
            return
        compression = "gzip" if value.size > 0 else None
        if compression is None:
            group.create_dataset(str(key), data=value)
        else:
            group.create_dataset(
                str(key),
                data=value,
                compression=compression,
                compression_opts=4,
                shuffle=True,
            )
        return

    if isinstance(value, (np.generic, bool, int, float, complex)):
        group.create_dataset(str(key), data=value)
        return

    raise TypeError(f"Unsupported value type for HDF5 serialization: key={key!r}, type={type(value)!r}")


def _read_value(node: h5py.Group | h5py.Dataset) -> Any:
    if isinstance(node, h5py.Group):
        kind = str(node.attrs.get("kind", ""))
        if kind == _DICT_KIND:
            return {str(key): _read_value(node[key]) for key in node.keys()}
        raise TypeError(f"Unsupported HDF5 group kind {kind!r}")

    value = node[()]
    if isinstance(value, np.ndarray) and value.dtype.kind == "S":
        return value.astype(str)
    if isinstance(value, np.ndarray) and value.dtype.kind == "O":
        if value.size == 0:
            return value
        if all(isinstance(item, (bytes, np.bytes_)) for item in value.flat):
            return np.asarray([item.decode("utf-8") for item in value.flat], dtype=str).reshape(value.shape)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _attr_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
