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
from pyspark.rdd import RDD
from pyspark.sql.types import LongType, StructField, StructType
from pyspark.sql import Row
import pyspark.sql as spark

from dupegrouper.constants import CANONICAL_ID, PYSPARK_TYPES
from dupegrouper.types import DataFrameLike, SeriesLike


# TYPES


T = TypeVar("T")


# BASE


class Frame(ABC, Generic[T]):

    def __init__(self, df: T):
        self._df: T = df

    def unwrap(self) -> T:
        return self._df

    def __getattr__(self, name: str) -> Any:
        """Delegation: use ._df without using property explicitely.

        So, the use of Self even with no attribute returns ._df attribute.
        Therefore calling Self == call Self._df. This is useful as it makes the
        API more concise in other modules.

        For example, as the Duped class attribute ._df is an instance of this
        class, it avoids having to do Duped()._df._df to access the actual
        dataframe.
        """
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
        self._df: pd.DataFrame = self._add_canonical_id(df, id)
        self._id = id
        # TODO validate that id exists in the DF!

    @staticmethod
    @override
    def _add_canonical_id(df: pd.DataFrame, id: str | None = None) -> pd.DataFrame:
        has_canonical: bool = CANONICAL_ID in df.columns
        id_is_canonical: bool = id == CANONICAL_ID

        if has_canonical:
            if id:
                if id_is_canonical:
                    return df
                # overwrite with id
                return df.assign(**{CANONICAL_ID: df[id]})
            #TODO: need a warning here. user needs to know that they should pass id="canonical_id"
            return df
        if id:
            # write new with id
            return df.assign(**{CANONICAL_ID: df[id]})
        # write new auto-incrementing
        return df.assign(**{CANONICAL_ID: pd.RangeIndex(start=0, stop=len(df))})

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
        self._df: pl.DataFrame = self._add_canonical_id(df, id)
        self._id = id
        # TODO validate that id exists in the DF!

    @staticmethod
    @override
    def _add_canonical_id(df: pl.DataFrame, id: str | None = None) -> pl.DataFrame:
        has_canonical: bool = CANONICAL_ID in df.columns
        id_is_canonical: bool = id == CANONICAL_ID

        if has_canonical:
            if id:
                if id_is_canonical:
                    return df
                # overwrite with id
                return df.with_columns(df[id].alias(CANONICAL_ID))
            return df
        if id:
            # write new with id
            return df.with_columns(df[id].alias(CANONICAL_ID))
        # write new auto-incrementing
        return df.with_columns(pl.arange(0, len(df)).alias(CANONICAL_ID))
        

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

    ERR_MSG = "Method is available for spark RDD, not spark DataFrame"

    def __init__(self, df: spark.DataFrame, id: str | None = None, is_init: bool = True):
        super().__init__(df)
        if is_init:
            self._df: RDD[Row] = self._add_canonical_id(df, id)
        else:
            self._df: spark.DataFrame = df
        self._id = id
        self._schema = self._get_schema(df)

    @staticmethod
    @override
    def _add_canonical_id(df: spark.DataFrame, id: str | None = None) -> RDD[Row]:
        has_canonical: bool = CANONICAL_ID in df.columns
        id_is_canonical: bool = id == CANONICAL_ID

        df = df.select("*")  # TODO move to __init__

        if has_canonical:
            if id:
                if id_is_canonical:
                    return df
                # overwrite with id
                df: spark.DataFrame = df.drop(CANONICAL_ID)
                return df.rdd.mapPartitions(
                    lambda partition: [Row(**{**row.asDict(), CANONICAL_ID: row[id]}) for row in partition]
                )
            return df
        if id:
            # write new with id
            return df.rdd.mapPartitions(
                lambda partition: [Row(**{**row.asDict(), CANONICAL_ID: row[id]}) for row in partition]
            )
        # write new auto-incrementing
        return df.rdd.zipWithIndex().mapPartitions(
            lambda partition: [Row(**{**row.asDict(), CANONICAL_ID: idx}) for row, idx in partition]
        )

    def _get_schema(self, df: SparkDF) -> StructType:
        fields = df.schema.fields
        if CANONICAL_ID in df.columns:
            return StructType(fields)

        if self._id:
            id_type = PYSPARK_TYPES.get(dict(df.dtypes).get(self._id))
        else:
            id_type = LongType()  # auto-incremental is numeric
        fields += [StructField(CANONICAL_ID, id_type, True)]
        return StructType(fields)

    # SPARK API WRAPPERS:

    @override
    def put_col(self):
        raise NotImplementedError(self.ERR_MSG)

    @override
    def get_col(self):
        raise NotImplementedError(self.ERR_MSG)

    @override
    def get_cols(self):
        raise NotImplementedError(self.ERR_MSG)


@final
class SparkRows(Frame[list[spark.Row]]):
    """Lower level DataFrame wrapper per partition i.e. list of Rows

    Can be emulated by operating on a collected pyspark dataframe i.e.
    df.collect()
    """

    def __init__(self, df: list[spark.Row], id: str):
        super().__init__(df)
        del id
        self._df = df

    @staticmethod
    @override
    def _add_canonical_id(df: list[spark.Row], id: str | None) -> list[spark.Row]:
        # TODO: consider `NotImplementedError`
        del df, id  # unused
        pass

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
    return SparkRows(df, id)


# ACCESSIBLE TYPES


LocalDF: TypeAlias = PandasDF | PolarsDF | SparkRows
DF = TypeVar("DF", SparkDF, LocalDF)
