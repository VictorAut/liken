"""
@private
ABC for wrapped dataframe interfaces
"""

from __future__ import annotations

from functools import singledispatch
from typing import Any, Generic, Protocol, Self, TypeAlias, TypeVar, final
import warnings

import numpy as np
import pandas as pd
import polars as pl
import pyspark.sql as spark
from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql.types import LongType, StructField, StructType
from typing_extensions import override

from dupegrouper.constants import CANONICAL_ID, PYSPARK_TYPES
from dupegrouper.types import DataFrameLike

# TYPES


T = TypeVar("T")


# BASE


class Frame(Generic[T]):

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


# CANONICAL ID


class AddsCanonical(Protocol):
    def _df_as_is(self, df): ...
    def _df_overwrite_id(self, df, id: str): ...
    def _df_copy_id(self, df, id: str): ...
    def _df_autoincrement_id(self, df): ...


class CanonicalIdMixin:

    def _add_canonical_id(self: AddsCanonical, df, id: str | None):
        has_canonical: bool = CANONICAL_ID in df.columns
        id_is_canonical: bool = id == CANONICAL_ID

        if has_canonical:
            if id:
                if id_is_canonical:
                    return self._df_as_is(df)
                # overwrite with id
                return self._df_overwrite_id(df, id)
            warnings.warn(
                f"Canonical ID '{CANONICAL_ID}' already exists. Pass '{CANONICAL_ID}' to `id` arg for consistency",
                category=UserWarning,
            )
            return self._df_as_is(df)
        if id:
            # write new with id
            return self._df_copy_id(df, id)
        # write new auto-incrementing
        return self._df_autoincrement_id(df)


# WRAPPERS


@final
class PandasDF(Frame[pd.DataFrame], CanonicalIdMixin):

    def __init__(self, df: pd.DataFrame, id: str | None = None):
        self._df: pd.DataFrame = self._add_canonical_id(df, id)
        self._id = id
        # TODO validate that id exists in the DF!

    def _df_as_is(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _df_overwrite_id(self, df: pd.DataFrame, id: str) -> pd.DataFrame:
        return df.assign(**{CANONICAL_ID: df[id]})

    def _df_copy_id(self, df: pd.DataFrame, id: str) -> pd.DataFrame:
        return df.assign(**{CANONICAL_ID: df[id]})

    def _df_autoincrement_id(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(**{CANONICAL_ID: pd.RangeIndex(start=0, stop=len(df))})

    # WRAPPER METHODS:

    def put_col(self, column: str, array) -> Self:
        self._df = self._df.assign(**{column: array})
        return self

    def get_col(self, column: str) -> pd.Series:
        return self._df[column]

    def get_cols(self, columns: tuple[str, ...]) -> pd.DataFrame:
        return self._df[list(columns)]


@final
class PolarsDF(Frame[pl.DataFrame], CanonicalIdMixin):

    def __init__(self, df: pl.DataFrame, id: str | None = None):
        self._df: pl.DataFrame = self._add_canonical_id(df, id)
        self._id = id
        # TODO validate that id exists in the DF!

    def _df_as_is(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def _df_overwrite_id(self, df: pl.DataFrame, id: str) -> pl.DataFrame:
        return df.with_columns(df[id].alias(CANONICAL_ID))

    def _df_copy_id(self, df: pl.DataFrame, id: str) -> pl.DataFrame:
        return df.with_columns(df[id].alias(CANONICAL_ID))

    def _df_autoincrement_id(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.arange(0, len(df)).alias(CANONICAL_ID))

    # WRAPPER METHODS:

    def put_col(self, column: str, array) -> Self:
        array = pl.Series(array)  # important; allow list to be assigned to column
        self._df = self._df.with_columns(**{column: array})
        return self

    def get_col(self, column: str) -> pl.Series:
        return self._df.get_column(column)

    def get_cols(self, columns: tuple[str, ...]) -> pl.DataFrame:
        return self._df.select(columns)


SparkObject: TypeAlias = spark.DataFrame | RDD[Row]


@final
class SparkDF(Frame[SparkObject], CanonicalIdMixin):

    ERR_MSG = "Method is available for spark RDD, not spark DataFrame"

    def __init__(
        self,
        df: spark.DataFrame,
        id: str | None = None,
        is_init: bool = True,
    ):
        # new spark plan
        df = df.select("*")

        self._df: SparkObject
        if is_init:
            self._df = self._add_canonical_id(df, id)
        else:
            self._df = df

        self._id = id

    def _df_as_is(self, df: spark.DataFrame) -> RDD[Row]:
        self._schema = df.schema
        return df.rdd

    def _df_overwrite_id(self, df: spark.DataFrame, id: str) -> RDD[Row]:
        df_new: spark.DataFrame = df.drop(CANONICAL_ID)
        self._schema = self._new_schema(df_new, id)
        return df_new.rdd.mapPartitions(
            lambda partition: [Row(**{**row.asDict(), CANONICAL_ID: row[id]}) for row in partition]
        )

    def _df_copy_id(self, df: spark.DataFrame, id: str) -> RDD[Row]:
        self._schema = self._new_schema(df, id)
        return df.rdd.mapPartitions(
            lambda partition: [Row(**{**row.asDict(), CANONICAL_ID: row[id]}) for row in partition]
        )

    def _df_autoincrement_id(self, df: spark.DataFrame) -> RDD[Row]:
        self._schema = self._new_schema(df)
        return df.rdd.zipWithIndex().mapPartitions(
            lambda partition: [Row(**{**row.asDict(), CANONICAL_ID: idx}) for row, idx in partition]
        )

    @staticmethod
    def _new_schema(df: spark.DataFrame, id: str | None = None) -> StructType:
        fields = df.schema.fields
        if id:
            dtype = dict(df.dtypes)[id]
            id_type = PYSPARK_TYPES[dtype]
        else:
            id_type = LongType()  # auto-incremental is numeric
        fields += [StructField(CANONICAL_ID, id_type, True)]
        return StructType(fields)

    @override
    def unwrap(self) -> spark.DataFrame:
        """Ensure the unwrapped dataframe is always an instance of DataFrame

        Permits the access of the base Duped class attribute dataframe to be
        returned as a DataFrame even if no canonicalisation has been applied
        yet. For example this would be needed if inspecting the dataframe as
        contained in an instance of Duped having yet to call the canonicalizer
        on the set of strategies"""
        if isinstance(self._df, RDD):
            return self._df.toDF()
        return self._df

    # WRAPPER METHODS:

    def put_col(self):
        raise NotImplementedError(self.ERR_MSG)

    def get_col(self):
        raise NotImplementedError(self.ERR_MSG)

    def get_cols(self):
        raise NotImplementedError(self.ERR_MSG)


@final
class SparkRows(Frame[list[spark.Row]]):
    def __init__(self, df: list[spark.Row]):
        self._df = df

    # WRAPPER METHODS:

    def put_col(self, column: str, array) -> Self:
        array = [i.item() if isinstance(i, np.generic) else i for i in array]
        self._df = [spark.Row(**{**row.asDict(), column: value}) for row, value in zip(self._df, array)]
        return self

    def get_col(self, column: str) -> list[Any]:
        return [row[column] for row in self._df]

    def get_cols(self, columns: tuple[str, ...]) -> list[list[Any]]:
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
def _(df, id: str | None = None) -> SparkDF:
    return SparkDF(df, id)


@wrap.register(list)
def _(df: list[spark.Row], id: str | None) -> SparkRows:
    del id
    return SparkRows(df)


# ACCESSIBLE TYPES


LocalDF: TypeAlias = PandasDF | PolarsDF | SparkRows
DF = TypeVar("DF", SparkDF, LocalDF)
