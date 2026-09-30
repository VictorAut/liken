"""Polars DataFrame wrapper"""

from __future__ import annotations

from typing import Self
from typing import final

import polars as pl
import pyarrow as pa

from liken.constants import CANONICAL_ID
from liken.core.wrapper import DF
from liken.core.wrapper import CanonicalIdMixin
from liken.types import Keep


@final
class PolarsDF(DF[pl.DataFrame], CanonicalIdMixin):
    """Polars DataFrame wrapper"""

    def __init__(self, df: pl.DataFrame, id: str | None = None):
        self._df: pl.DataFrame = self._add_canonical_id(df, id)
        self._id = id

    # CANONICAL ID HELPERS:

    def _df_as_is(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def _df_overwrite_id(self, df: pl.DataFrame, id: str) -> pl.DataFrame:
        return df.with_columns(df[id].alias(CANONICAL_ID))

    def _df_copy_id(self, df: pl.DataFrame, id: str) -> pl.DataFrame:
        return df.with_columns(df[id].alias(CANONICAL_ID))

    def _df_autoincrement_id(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.arange(0, len(df)).alias(CANONICAL_ID))

    def _column_labels_list(self, df: pl.DataFrame) -> list[str]:
        return df.columns

    # ARROW INTERFACES:

    def _get_col(self, column: str) -> pa.Array:
        return pa.array(self._df.get_column(column))

    def _get_cols(self, columns: tuple[str, ...]) -> pa.Table:
        return self._df.select(columns).to_arrow()

    # WRAPPER METHODS:

    def put_col(self, column: str, array: list) -> Self:
        array: pl.Series = pl.Series(array)  # IMPORTANT; allow list to be assigned to column
        self._df = self._df.with_columns(**{column: array})
        return self

    def drop_col(self, column: str) -> Self:
        self._df = self._df.drop(column)
        return self

    def drop_duplicates(self, keep: Keep) -> Self:
        self._df = self._df.unique(keep=keep, subset=CANONICAL_ID, maintain_order=True)
        return self

    # SYNTHETIC RECORD:

    def synthesize_record(self) -> pl.DataFrame:
        exprs = []

        for col in self._df.columns:
            if col == CANONICAL_ID:
                continue

            exprs.append(pl.col(col).drop_nulls().first().alias(col))

        return self._df.group_by(CANONICAL_ID).agg(exprs).sort(CANONICAL_ID)
