"""
@private
ABC for wrapped dataframe interfaces
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from functools import singledispatch
from typing_extensions import override
from typing import Any, final, Generic, Self, TypeAlias, TypeVar

import numpy as np
import pandas as pd
import polars as pl
import pyspark.sql as spark

from dupegrouper.constants import CANONICAL_ID
from dupegrouper.types import DataFrameLike, SeriesLike


# TYPES


T = TypeVar("T")


# BASE


class Frame(ABC, Generic[T]):

    def __init__(self, df: T):
        self._df: T = df

    def unwrap(self) -> T:
        return self._df

    # delegation: use ._df without using property explicitely

    def __getattr__(self, name: str) -> Any:
        return getattr(self._df, name)

    # Protocol:

    @staticmethod
    @abstractmethod
    def _add_canonical_id(df: T, id: str | None):
        pass

    @abstractmethod
    def put_col(self, column: str, array) -> Self:
        pass

    @abstractmethod
    def get_col(self, column: str) -> SeriesLike:
        pass

    @abstractmethod
    def get_cols(self, columns: tuple[str, ...]) -> T:
        pass


# WRAPPERS


@final
class PandasDF(Frame[pd.DataFrame]):

    def __init__(self, df: pd.DataFrame, id: str | None):
        super().__init__(df)
        self._df: pd.DataFrame = self._add_canonical_id(df)
        self._id = id

    @staticmethod
    @override
    def _add_canonical_id(df: pd.DataFrame, id: str | None = None) -> pd.DataFrame:
        return df.assign(**{CANONICAL_ID: pd.RangeIndex(start=1, stop=len(df) + 1)})

    # PANDAS API WRAPPERS:

    @override
    def put_col(self, column: str, array) -> Self:
        self._df = self._df.assign(**{column: array})
        return self

    @override
    def get_col(self, column: str) -> pd.Series:
        return self._df[column]

    @override
    def get_cols(self, columns: tuple[str, ...]) -> pd.DataFrame:
        return self._df[list(columns)]


@final
class PolarsDF(Frame[pl.DataFrame]):

    def __init__(self, df: pl.DataFrame, id: str | None):
        super().__init__(df)
        self._df: pl.DataFrame = self._add_canonical_id(df)
        self._id = id

    @staticmethod
    @override
    def _add_canonical_id(df: pl.DataFrame, id: str | None = None) -> pl.DataFrame:
        return df.with_columns(pl.arange(1, len(df) + 1).alias(CANONICAL_ID))

    # POLARS API WRAPPERS:

    @override
    def put_col(self, column: str, array) -> Self:
        array = pl.Series(array)  # important; allow list to be assigned to column
        self._df = self._df.with_columns(**{column: array})
        return self

    @override
    def get_col(self, column: str) -> pl.Series:
        return self._df.get_column(column)

    @override
    def get_cols(self, columns: tuple[str, ...]) -> pl.DataFrame:
        return self._df.select(columns)


@final
class SparkDF(Frame[spark.DataFrame]):

    err_msg = "Spark DataFrame methods are available per partition only, i.e. for lists of `pyspark.sql.Row`"

    def __init__(self, df: spark.DataFrame, id: str | None):
        super().__init__(df)
        del id  # Not implemented, input param there for API consistency

    @override
    def _add_canonical_id(self):
        raise NotImplementedError(self.err_msg)

    # SPARK API WRAPPERS:

    @override
    def put_col(self):
        raise NotImplementedError(self.err_msg)

    @override
    def get_col(self):
        raise NotImplementedError(self.err_msg)

    @override
    def get_cols(self):
        raise NotImplementedError(self.err_msg)


@final
class SparkRows(Frame[list[spark.Row]]):
    """Lower level DataFrame wrapper per partition i.e. list of Rows

    Can be emulated by operating on a collected pyspark dataframe i.e.
    df.collect()
    """

    def __init__(self, df: list[spark.Row], id: str):
        super().__init__(df)
        self._df: list[spark.Row] = self._add_canonical_id(df, id)

    @staticmethod
    @override
    def _add_canonical_id(df: list[spark.Row], id: str | None) -> list[spark.Row]:
        return [spark.Row(**{**row.asDict(), CANONICAL_ID: row[id]}) for row in df]

    # SPARK API WRAPPERS:

    @override
    def put_col(self, column: str, array) -> Self:
        array = [i.item() if isinstance(i, np.generic) else i for i in array]
        self._df = [spark.Row(**{**row.asDict(), column: value}) for row, value in zip(self._df, array)]
        return self

    @override
    def get_col(self, column: str) -> list[Any]:
        return [row[column] for row in self._df]

    @override
    def get_cols(self, columns: tuple[str, ...]) -> list[list[Any]]:  # type: ignore
        return [[row[c] for c in columns] for row in self._df]


# DISPATCHER:


@singledispatch
def wrap(df: DataFrameLike, id: str | None = None):
    """
    Dispatch the dataframe to the appropriate wrapping handler.

    Args:
        df: The dataframe to dispatch to the appropriate handler.

    Returns:
        Frame, a DataFrame wrapped with a uniform interface.

    Raises:
        NotImplementedError
    """
    del id  # Unused
    raise NotImplementedError(f"Unsupported data frame: {type(df)}")


@wrap.register(pd.DataFrame)
def _(df, id: str | None = None) -> PandasDF:
    return PandasDF(df, id)


@wrap.register(pl.DataFrame)
def _(df, id: str | None = None) -> PolarsDF:
    return PolarsDF(df, id)


@wrap.register(spark.DataFrame)
def _(df, id: str) -> SparkDF:
    return SparkDF(df, id)


@wrap.register(list)
def _(df: list[spark.Row], id: str) -> SparkRows:
    """As lists can be large: `all` membership is `Row` is not validated"""
    return SparkRows(df, id)


# ACCESSIBLE TYPES


LocalDF: TypeAlias = PandasDF | PolarsDF | SparkRows
DF = TypeVar("DF", SparkDF, LocalDF)