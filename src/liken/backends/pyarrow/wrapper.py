"""PyArrow DataFrame wrapper"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Self
from typing import final

import pyarrow as pa

from liken.core.missing import is_missing
from liken.core.wrapper import DF
from liken.core.wrapper import CanonicalIdMixin


if TYPE_CHECKING:
    from liken.types import Keep


@final
class ArrowDF(DF[pa.Table], CanonicalIdMixin):
    """PyArrow Table wrapper"""

    def __init__(self, df: pa.Table, source_id: str | None = None, *, canonical_id: str):
        self._df: pa.Table = self._add_canonical_id(df, source_id, canonical_id)
        self._source_id = source_id
        self._canonical_id = canonical_id

    # CANONICAL ID HELPERS:

    def _df_as_is(self, df: pa.Table) -> pa.Table:
        return df

    def _df_overwrite_id(self, df: pa.Table, source_id: str, canonical_id: str) -> pa.Table:
        return df.set_column(df.column_names.index(canonical_id), canonical_id, df.column(source_id))

    def _df_copy_id(self, df: pa.Table, source_id: str, canonical_id: str) -> pa.Table:
        return df.append_column(canonical_id, df.column(source_id))

    def _df_autoincrement_id(self, df: pa.Table, canonical_id: str) -> pa.Table:
        return df.append_column(canonical_id, pa.array(range(len(df)), type=pa.int64()))

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
        canonical = self._df.column(self._canonical_id).to_pylist()

        # keep "first": record only the first occurrence of each id.
        # keep "last": overwrite with each later occurrence.
        keep_last = keep == "last"
        kept: dict[Any, int] = {}
        for index, value in enumerate(canonical):
            key = None if is_missing(value) else value
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
        canonical_id = self._canonical_id
        canonical = df.column(canonical_id)
        canonical_type = canonical.type

        groups: dict[Any, list[int]] = {}
        for index, value in enumerate(canonical.to_pylist()):
            # Missing canonical ids carry no group; the pandas reference drops them.
            if is_missing(value):
                continue
            groups.setdefault(value, []).append(index)

        keys = sorted(groups)
        data_names = [name for name in df.column_names if name != canonical_id]

        columns: list[pa.Array] = [pa.array(keys, type=canonical_type)]
        for name in data_names:
            values = df.column(name).to_pylist()
            merged: list[object] = [_first_non_null(values, groups[key]) for key in keys]
            columns.append(pa.array(merged, type=df.column(name).type))

        return pa.Table.from_arrays(columns, names=[canonical_id, *data_names])


def _first_non_null(values: list[Any], indices: list[int]) -> Any:
    for index in indices:
        if not is_missing(values[index]):
            return values[index]
    return None
