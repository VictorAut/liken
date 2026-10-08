"""Exploratory duplicate-rate profiling"""

from __future__ import annotations

import copy
import random
from typing import TYPE_CHECKING
from typing import Any
from typing import Final
from typing import cast

import pyarrow as pa

from liken.constants import INVALID_EXPLORE_BACKEND
from liken.core.dispatcher import get_backend
from liken.core.dispatcher import wrap
from liken.dedupers.exact import exact
from liken.dedupers.fuzzy import fuzzy
from liken.validators import validate_explore_column_exists


if TYPE_CHECKING:
    from liken.core.backend import Backend
    from liken.core.deduper import BaseDeduper
    from liken.core.deduper import ThresholdDeduper
    from liken.core.wrapper import DF
    from liken.types import UserDataFrame


DEFAULT_EXPLORE_THRESHOLDS: Final[list[float]] = [0.5, 0.75, 0.9, 0.95, 0.99]

_SUPPORTED_BACKENDS: Final[frozenset[str]] = frozenset({"pandas", "polars", "modin", "pyarrow"})

_EXACT_LABEL: Final[str] = "exact"
_METRIC_LABEL: Final[str] = "metric"


def run_explore(
    df: UserDataFrame,
    columns: list[str] | dict[str, BaseDeduper],
    thresholds: list[float],
    frac: float,
) -> UserDataFrame:
    """Compute a describe-like duplicate-rate table. See `Dedupe.explore`."""

    backend: Backend = get_backend(df)

    if backend.name not in _SUPPORTED_BACKENDS:
        raise ValueError(INVALID_EXPLORE_BACKEND.format(backend.name))

    df_columns: list[str] = _column_labels(df, backend.name)

    if isinstance(columns, dict):
        col_names: list[str] = list(columns.keys())
        base_dedupers: dict[str, BaseDeduper] = dict(columns)
    else:
        col_names = list(columns)
        base_dedupers = {col: fuzzy() for col in col_names}

    for col in col_names:
        validate_explore_column_exists(col, df_columns)

    # Sampling returns a new frame and wrapping copies, so the user's
    # dataframe is never mutated.
    wdf: DF = wrap(_sample(df, backend.name, frac), id=None)

    rows: list[tuple[str | float, ...]] = []

    exact_row: list[str | float] = [_EXACT_LABEL]
    exact_row.extend(_duplicate_rate(exact(), wdf, col) for col in col_names)
    rows.append(tuple(exact_row))

    for threshold in thresholds:
        row: list[str | float] = [str(threshold)]
        row.extend(
            _duplicate_rate(
                _at_threshold(base_dedupers[col], threshold),
                wdf,
                col,
            )
            for col in col_names
        )
        rows.append(tuple(row))

    result: UserDataFrame = backend.create_df(data=rows, schema=[_METRIC_LABEL, *col_names])

    if backend.name in ("pandas", "modin"):
        result = cast("Any", result).set_index(_METRIC_LABEL)

    return result


def _column_labels(df: UserDataFrame, backend_name: str) -> list[str]:
    """Column labels of `df`.

    `pyarrow.Table.columns` returns the column arrays, not the labels.
    """
    if backend_name == "pyarrow":
        return list(cast("pa.Table", df).column_names)
    return list(cast("Any", df).columns)


def _sample(df: UserDataFrame, backend_name: str, frac: float) -> UserDataFrame:
    """Return a random `frac` of rows (a new frame), or the frame as-is."""
    if frac == 1.0:
        return df
    if backend_name == "polars":
        return cast("UserDataFrame", cast("Any", df).sample(fraction=frac))
    if backend_name == "pyarrow":
        # pa.Table has no sample; take a random subset of row indices. A typed
        # index array keeps an empty take from failing on a null-typed
        # empty indices argument.
        n = cast("pa.Table", df).num_rows
        k = round(frac * n)
        indices = sorted(random.sample(range(n), k))
        return cast("UserDataFrame", cast("pa.Table", df).take(pa.array(indices, type=pa.int64())))
    return cast("UserDataFrame", cast("Any", df).sample(frac=frac))


def _duplicate_rate(deduper: BaseDeduper, wdf: DF, column: str) -> float:
    """Fraction of rows that are redundant duplicates under `deduper`."""
    uf, n = deduper.set_frame(wdf).build_union_find(column, [])
    if n == 0:
        return 0.0
    n_groups = len({uf[i] for i in range(n)})
    return (n - n_groups) / n


def _at_threshold(base: BaseDeduper, threshold: float) -> BaseDeduper:
    """A copy of `base` set to `threshold`.

    All built-in similarity dedupers read `self._threshold` at run time, so a
    shallow copy with the threshold reassigned yields a correct variant without
    reconstructing the deduper.
    """
    deduper = cast("ThresholdDeduper", copy.copy(base))
    deduper._threshold = threshold
    return deduper
