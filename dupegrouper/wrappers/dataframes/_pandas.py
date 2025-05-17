"""Defines methods after Pandas API"""

from __future__ import annotations
from typing_extensions import override
import typing

import pandas as pd

from dupegrouper.definitions import GROUP_ID
from dupegrouper.wrappers.dataframe import WrappedDataFrame


class WrappedPandasDataFrame(WrappedDataFrame):

    def __init__(self, df: pd.DataFrame, id: str | None):
        super().__init__(df)
        self._df: pd.DataFrame = self._add_group_id(df)
        self._id = id

    @staticmethod
    @override
    def _add_group_id(df) -> pd.DataFrame:
        return df.assign(**{GROUP_ID: pd.RangeIndex(start=1, stop=len(df) + 1)})

    # PANDAS API WRAPPERS:

    @override
    def put_col(self, column: str, array) -> typing.Self:
        self._df = self._df.assign(**{column: array})
        return self

    @override
    def get_col(self, column: str) -> pd.Series:
        return self._df[column]

    @override
    def map_dict(self, column: str, mapping: dict) -> pd.Series:
        return self.get_col(column).map(mapping)

    @override
    def drop_col(self, column: str) -> typing.Self:
        self._df = self._df.drop(columns=column)
        return self

    @staticmethod
    @override
    def fill_na(series: pd.Series, array: pd.Series) -> pd.Series:
        return series.fillna(array)
