"""Utilities for persisting benchmark dataclass results to HDF5."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import h5py
import numpy as np

_RESULT_GROUP = "result"
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
        root = h5.create_group(_RESULT_GROUP)
        for f in fields(result):
            _write_value(root, f.name, getattr(result, f.name))


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
        schema_version = _attr_text(h5.attrs.get("schema_version", ""))
        if schema_version and schema_version != _SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {schema_version!r}; expected {_SCHEMA_VERSION!r}"
            )

        benchmark_name = _attr_text(h5.attrs.get("benchmark_name", ""))
        if benchmark_name and benchmark_name != expected_benchmark_name:
            raise ValueError(
                f"Results file benchmark {benchmark_name!r} does not match "
                f"expected {expected_benchmark_name!r}"
            )
        result_class = _attr_text(h5.attrs.get("result_class", ""))
        expected_class = result_type.__name__
        if result_class and result_class != expected_class:
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
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _attr_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
