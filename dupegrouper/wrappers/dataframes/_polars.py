"""Defines methods after Polars API"""

from __future__ import annotations
from typing_extensions import override
import typing

import polars as pl

from dupegrouper.definitions import GROUP_ID
from dupegrouper.wrappers.dataframe import WrappedDataFrame


class WrappedPolarsDataFrame(WrappedDataFrame):

    def __init__(self, df: pl.DataFrame, id: str | None):
        super().__init__(df)
        self._df: pl.DataFrame = self._add_group_id(df)
        self._id = id

    @staticmethod
    @override
    def _add_group_id(df) -> pl.DataFrame:
        return df.with_columns(pl.arange(1, len(df) + 1).alias(GROUP_ID))

    # POLARS API WRAPPERS:

    @override
    def put_col(self, column: str, array) -> typing.Self:
        self._df = self._df.with_columns(**{column: array})
        return self

    @override
    def get_col(self, column: str) -> pl.Series:
        return self._df.get_column(column)

    @override
    def map_dict(self, column: str, mapping: dict) -> pl.Series:
        return self.get_col(column).replace_strict(mapping, default=None)

    @override
    def drop_col(self, column: str) -> typing.Self:
        self._df = self._df.drop(column)
        return self

    @staticmethod
    @override
    def fill_na(series: pl.Series, array: pl.Series) -> pl.Series:
        return series.fill_null(array)
