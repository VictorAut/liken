"""Defines methods after Spark API"""

from __future__ import annotations
from typing_extensions import override
import typing

import numpy as np
from pyspark.sql import DataFrame, Row

from dupegrouper.definitions import GROUP_ID
from dupegrouper.wrappers.dataframe import WrappedDataFrame


class WrappedSparkDataFrame(WrappedDataFrame):

    not_implemented = "Spark DataFrame methods are available per partition only, i.e. for lists of `pyspark.sql.Row`"

    def __init__(self, df: DataFrame, id: str | None):
        super().__init__(df)
        del id  # Not implemented, input param there for API consistency

    @override
    def _add_group_id(self):
        raise NotImplementedError(self.not_implemented)  # pragma: no cover

    # SPARK API WRAPPERS:

    @override
    def put_col(self):
        raise NotImplementedError(self.not_implemented)

    @override
    def get_col(self):
        raise NotImplementedError(self.not_implemented)

    @override
    def map_dict(self):
        raise NotImplementedError(self.not_implemented)

    @override
    def drop_col(self):
        raise NotImplementedError(self.not_implemented)

    @override
    def fill_na(self):
        raise NotImplementedError(self.not_implemented)


class WrappedSparkRows(WrappedDataFrame):
    """Lower level DataFrame wrapper per partition i.e. list of Rows

    Can be emulated by operating on a collected pyspark dataframe i.e.
    df.collect()
    """

    def __init__(self, df: list[Row], id: str):
        super().__init__(df)
        self._df: list[Row] = self._add_group_id(df, id)

    @staticmethod
    @override
    def _add_group_id(df: list[Row], id: str) -> list[Row]:  # type: ignore[override]
        return [Row(**{**row.asDict(), GROUP_ID: row[id]}) for row in df]

    # SPARK API WRAPPERS:

    @override
    def put_col(self, column: str, array) -> typing.Self:
        array = [i.item() if isinstance(i, np.generic) else i for i in array]
        self._df = [Row(**{**row.asDict(), column: value}) for row, value in zip(self._df, array)]
        return self

    @override
    def get_col(self, column: str) -> list[typing.Any]:
        return [row[column] for row in self._df]

    @override
    def map_dict(self, column: str, mapping: dict) -> list[typing.Any]:
        return [mapping.get(row[column]) for row in self._df]

    @override
    def drop_col(self, column: str) -> typing.Self:
        self._df = [Row(**{k: v for k, v in row.asDict().items() if k != column}) for row in self._df]
        return self

    @staticmethod
    @override
    def fill_na(series: list, array: list) -> list:
        return [i[-1] if not i[0] else i[0] for i in zip(series, array)]
