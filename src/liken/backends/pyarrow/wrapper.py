"""PyArrow DataFrame wrapper"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import Any
from typing import Self
from typing import final

import pyarrow as pa

from liken.constants import CANONICAL_ID
from liken.core.wrapper import DF
from liken.core.wrapper import CanonicalIdMixin


if TYPE_CHECKING:
    from liken.types import Keep


@final
class ArrowDF(DF[pa.Table], CanonicalIdMixin):
    """PyArrow Table wrapper"""

    def __init__(self, df: pa.Table, id: str | None = None):
        self._df: pa.Table = self._add_canonical_id(df, id)
        self._id = id

    # CANONICAL ID HELPERS:

    def _df_as_is(self, df: pa.Table) -> pa.Table:
        return df

    def _df_overwrite_id(self, df: pa.Table, id: str) -> pa.Table:
        return df.set_column(df.column_names.index(CANONICAL_ID), CANONICAL_ID, df.column(id))

    def _df_copy_id(self, df: pa.Table, id: str) -> pa.Table:
        return df.append_column(CANONICAL_ID, df.column(id))

    def _df_autoincrement_id(self, df: pa.Table) -> pa.Table:
        return df.append_column(CANONICAL_ID, pa.array(range(len(df)), type=pa.int64()))

    def _column_labels_list(self, df: pa.Table) -> list[str]:
        return df.column_names

    # ARROW INTERFACES:

    def _get_col(self, column: str) -> pa.Array:
        col = self._df.column(column)
        chunks = col.chunks
        if not chunks:
            return pa.array([], type=col.type)
        if len(chunks) == 1:
            return chunks[0]
        return pa.concat_arrays(chunks)

    def _get_cols(self, columns: tuple[str, ...]) -> pa.Table:
        return self._df.select(list(columns))

    # WRAPPER METHODS:

    def put_col(self, column: str, array: list) -> Self:
        col = pa.array(array)
        if column in self._df.column_names:
            index = self._df.column_names.index(column)
            self._df = self._df.set_column(index, column, col)
        else:
            self._df = self._df.append_column(column, col)
        return self

    def drop_col(self, column: str) -> Self:
        self._df = self._df.drop_columns([column])
        return self

    def drop_duplicates(self, keep: Keep) -> Self:
        canonical = self._df.column(CANONICAL_ID).to_pylist()

        # keep "first": record only the first occurrence of each id.
        # keep "last": overwrite with each later occurrence.
        keep_last = keep == "last"
        kept: dict[Any, int] = {}
        for index, value in enumerate(canonical):
            key = None if _is_missing(value) else value
            if keep_last or key not in kept:
                kept[key] = index

        indices = sorted(kept.values())
        # A typed index array keeps an empty take from failing on a null-typed
        # empty indices argument.
        self._df = self._df.take(pa.array(indices, type=pa.int64()))
        return self

    # SYNTHETIC RECORD:

    def synthesize_record(self) -> pa.Table:
        df = self._df
        canonical = df.column(CANONICAL_ID)
        canonical_type = canonical.type

        groups: dict[Any, list[int]] = {}
        for index, value in enumerate(canonical.to_pylist()):
            # Missing canonical ids carry no group; the pandas reference drops them.
            if _is_missing(value):
                continue
            groups.setdefault(value, []).append(index)

        keys = sorted(groups)
        data_names = [name for name in df.column_names if name != CANONICAL_ID]

        columns: list[pa.Array] = [pa.array(keys, type=canonical_type)]
        for name in data_names:
            values = df.column(name).to_pylist()
            merged: list[object] = [_first_non_null(values, groups[key]) for key in keys]
            columns.append(pa.array(merged, type=df.column(name).type))

        return pa.Table.from_arrays(columns, names=[CANONICAL_ID, *data_names])


def _is_missing(value: object) -> bool:
    """True for nulls and float NaNs, matching how pandas treats missing values."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def _first_non_null(values: list[Any], indices: list[int]) -> Any:
    for index in indices:
        if not _is_missing(values[index]):
            return values[index]
    return None
