"""ABC for wrapped dataframe interfaces"""

from __future__ import annotations
from abc import ABC, abstractmethod
from functools import singledispatch
from typing_extensions import override
import typing

import numpy as np
import pandas as pd
import polars as pl
from pyspark.sql import (
    DataFrame as SparkDataFrame,  # avoid name clash
    Row,
)

from dupegrouper.definitions import CANONICAL_ID, DataFrameLike, SeriesLike


class WrappedDataFrame(ABC):
    """Container class for a dataframe and associated methods

    At runtime any instance of this class will also be a data container of the
    dataframe. The abstractmethods defined here are all the required
    implementations needed
    """

    def __init__(self, df: DataFrameLike):
        self._df: DataFrameLike = df

    def unwrap(self) -> DataFrameLike:
        return self._df

    @staticmethod
    @abstractmethod
    def _add_canonical_id(df: DataFrameLike):
        """Return a dataframe with a group id column"""
        pass  # pragma: no cover

    # DATAFRAME `LIBRARY` WRAPPERS:

    @abstractmethod
    def put_col(self, column: str, array) -> typing.Self:
        """assign i.e. write a column with array-like data

        No return; `_df` is updated
        """
        pass  # pragma: no cover

    @abstractmethod
    def get_col(self, column: str) -> SeriesLike:
        """Return a column array-like of data"""
        pass  # pragma: no cover

    @abstractmethod
    def get_cols(self, columns: typing.Iterable[str]) -> DataFrameLike:
        """Return columns dataframe-like of data"""
        pass  # pragma: no cover


    # THIN TRANSPARENCY DELEGATION

    def __getattr__(self, name: str) -> typing.Any:
        return getattr(self._df, name)


class WrappedPandasDataFrame(WrappedDataFrame):

    def __init__(self, df: pd.DataFrame, id: str | None):
        super().__init__(df)
        self._df: pd.DataFrame = self._add_canonical_id(df)
        self._id = id

    @staticmethod
    @override
    def _add_canonical_id(df) -> pd.DataFrame:
        return df.assign(**{CANONICAL_ID: pd.RangeIndex(start=1, stop=len(df) + 1)})

    # PANDAS API WRAPPERS:

    @override
    def put_col(self, column: str, array) -> typing.Self:
        self._df = self._df.assign(**{column: array})
        return self

    @override
    def get_col(self, column: str) -> pd.Series:
        return self._df[column]

    @override
    def get_cols(self, columns: typing.Iterable[str]) -> pd.DataFrame:
        return self._df[list(columns)]


class WrappedPolarsDataFrame(WrappedDataFrame):

    def __init__(self, df: pl.DataFrame, id: str | None):
        super().__init__(df)
        self._df: pl.DataFrame = self._add_canonical_id(df)
        self._id = id

    @staticmethod
    @override
    def _add_canonical_id(df) -> pl.DataFrame:
        return df.with_columns(pl.arange(1, len(df) + 1).alias(CANONICAL_ID))

    # POLARS API WRAPPERS:

    @override
    def put_col(self, column: str, array) -> typing.Self:
        array = pl.Series(array)  # important; allow list to be assigned to column
        self._df = self._df.with_columns(**{column: array})
        return self

    @override
    def get_col(self, column: str) -> pl.Series:
        return self._df.get_column(column)

    @override
    def get_cols(self, columns: typing.Iterable[str]) -> pl.DataFrame:
        return self._df.select(columns)


class WrappedSparkDataFrame(WrappedDataFrame):

    not_implemented = "Spark DataFrame methods are available per partition only, i.e. for lists of `pyspark.sql.Row`"

    def __init__(self, df: DataFrame, id: str | None):
        super().__init__(df)
        del id  # Not implemented, input param there for API consistency

    @override
    def _add_canonical_id(self):
        raise NotImplementedError(self.not_implemented)  # pragma: no cover

    # SPARK API WRAPPERS:

    @override
    def put_col(self):
        raise NotImplementedError(self.not_implemented)

    @override
    def get_col(self):
        raise NotImplementedError(self.not_implemented)

    @override
    def get_cols(self):
        raise NotImplementedError(self.not_implemented)


class WrappedSparkRows(WrappedDataFrame):
    """Lower level DataFrame wrapper per partition i.e. list of Rows

    Can be emulated by operating on a collected pyspark dataframe i.e.
    df.collect()
    """

    def __init__(self, df: list[Row], id: str):
        super().__init__(df)
        self._df: list[Row] = self._add_canonical_id(df, id)

    @staticmethod
    @override
    def _add_canonical_id(df: list[Row], id: str) -> list[Row]:  # type: ignore[override]
        return [Row(**{**row.asDict(), CANONICAL_ID: row[id]}) for row in df]

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
    def get_cols(self, columns: typing.Iterable[str]) -> list[list[typing.Any]]:
        return [[row[c] for c in columns] for row in self._df]


# WRAP DATAFRAME DISPATCHER:


@singledispatch
def wrap(df: DataFrameLike, id: str | None = None) -> WrappedDataFrame:
    """
    Dispatch the dataframe to the appropriate wrapping handler.

    Args:
        df: The dataframe to dispatch to the appropriate handler.

    Returns:
        WrappedDataFrame, a DataFrame wrapped with a uniform interface.

    Raises:
        NotImplementedError
    """
    del id  # Unused
    raise NotImplementedError(f"Unsupported data frame: {type(df)}")


@wrap.register(pd.DataFrame)
def _(df, id: str | None = None):
    return WrappedPandasDataFrame(df, id)


@wrap.register(pl.DataFrame)
def _(df, id: str | None = None):
    return WrappedPolarsDataFrame(df, id)


@wrap.register(SparkDataFrame)
def _(df, id: str | None = None):
    return WrappedSparkDataFrame(df, id)


@wrap.register(list)
def _(df: list[Row], id: str):
    """As lists can be large: `all` membership is `Row` is not validated"""
    return WrappedSparkRows(df, id)
