"""
This module provides wrappers to allow for a uniform interface across different
backends. The backends covered are:
    - Pandas
    - Polars
    - Spark DataFrames
    - Spark RDDs
    - Spark Rows

Whilst Pandas and Polars wrappers are similarly wrapped, note the following:
- Spark Rows inherits the majority of functionality related to getting
    columns, puting columns, fill na etc
- Conversely, Spark DataFrames take care of adding canonical IDs

Additional Points regarding Spark. Upon initialising the public API with a
Spark DataFrame, the wrapper will call the SparkDF class which will create
canonical IDs. However the output to this is RDDs which are then processed
by the executor into Spark Rows which are dispatched to worker nodes. Spark
Rows can be fully recovered to a Spark DataFrame using the same SparkDF class.

TODO:
    - CanonicalIdMixin should be defined first when inherited
    - A full interface can then be defined
"""

# mypy: disable-error-code="no-redef"

from __future__ import annotations

import warnings
from collections.abc import Hashable
from functools import singledispatch
from typing import Any
from typing import Generic
from typing import Protocol
from typing import Self
from typing import TypeAlias
from typing import TypeVar
from typing import final

import numpy as np
import pandas as pd
import polars as pl
import pyspark.sql as spark
from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql.types import LongType
from pyspark.sql.types import StructField
from pyspark.sql.types import StructType
from typing_extensions import override

from liken._constants import CANONICAL_ID
from liken._constants import NA_PLACEHOLDER
from liken._constants import PYSPARK_TYPES
from liken._types import Columns
from liken._types import DataFrameLike
from liken._types import Keep


# TYPES


D = TypeVar("D")  # dataframe
S = TypeVar("S")  # Series


# BASE


class Frame(Generic[D, S]):
    """Base class defining a dataframe wrapper

    Defines inheritable methods as well as some of the interface

    TODO:
        - define a protocol interface
        - tighten generics
    """

    def __init__(self, df: D):
        self._df: D = df

    def unwrap(self) -> D:
        return self._df

    def __getattr__(self, name: str) -> Any:
        """Delegation: use ._df without using property explicitely.

        So, the use of Self even with no attribute returns ._df attribute.
        Therefore calling Self == call Self._df. This is useful as it makes the
        API more concise in other modules.

        For example, as the Dedupe class attribute ._df is an instance of this
        class, it avoids having to do Dedupe()._df._df to access the actual
        dataframe.
        """
        return getattr(self._df, name)

    def _get_col(self, column: str) -> S:
        del column
        raise NotImplementedError

    def _get_cols(self, columns: tuple[str, ...]) -> D:
        del columns
        raise NotImplementedError

    @staticmethod
    def _fill_na(series: S, value: str) -> S:
        del series, value
        raise NotImplementedError

    def get_array(self, columns: Columns, with_na: bool = False) -> np.ndarray:
        """Generalise the getting of a column, or columns of a df to an array.

        Handles single column and multicolumn. For instances of single column
        the initial column can initially be filled null placeholders, to allow
        for use my strategies. This is optional so that specific strategies
        that do care about nulls are not affected (e.g. IsNA).
        """
        if isinstance(columns, str):
            cols: S = self._get_col(columns)
            if with_na:
                return np.asarray(self._fill_na(cols, NA_PLACEHOLDER), dtype=object)
            return np.asarray(cols, dtype=object)
        return np.asarray(self._get_cols(columns), dtype=object)

    def get_canonical(self) -> np.ndarray:
        """convenience method for clean use"""
        return self.get_array(CANONICAL_ID)


# CANONICAL ID


class AddsCanonical(Protocol):
    """Mixin protocol"""

    def _df_as_is(self, df): ...
    def _df_overwrite_id(self, df, id: str): ...
    def _df_copy_id(self, df, id: str): ...
    def _df_autoincrement_id(self, df): ...


class CanonicalIdMixin(AddsCanonical):
    """Defines creation of canonical id upon wrapping a dataframe

    By default a canonical ID is an auto-incrementing numeric field, starting
    from zero.

    However, the canonical ID field can also be:
        - already present in the dataframe as "canonical_id"
        - copied from another "id" field

    In those other instances the resultant canonical id field can therefore
    also be a string field.
    """

    def _add_canonical_id(self, df, id: str | None):

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
class PandasDF(Frame[pd.DataFrame, pd.Series], CanonicalIdMixin):
    """Pandas DataFrame wrapper"""

    def __init__(self, df: pd.DataFrame, id: str | None = None):
        self._df: pd.DataFrame = self._add_canonical_id(df, id)
        self._id = id

    def _df_as_is(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _df_overwrite_id(self, df: pd.DataFrame, id: str) -> pd.DataFrame:
        return df.assign(**{CANONICAL_ID: df[id]})

    def _df_copy_id(self, df: pd.DataFrame, id: str) -> pd.DataFrame:
        return df.assign(**{CANONICAL_ID: df[id]})

    def _df_autoincrement_id(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(**{CANONICAL_ID: pd.RangeIndex(start=0, stop=len(df))})

    # WRAPPER METHODS:

    @staticmethod
    @override
    def _fill_na(series: pd.Series, value: str) -> pd.Series:
        return series.fillna(value)

    def _get_col(self, column: str) -> pd.Series:
        return self._df[column]

    def _get_cols(self, columns: tuple[str, ...]) -> pd.DataFrame:
        return self._df[list(columns)]

    def put_col(self, column: str, array) -> Self:
        self._df = self._df.assign(**{column: array})
        return self

    def drop_col(self, column: str) -> Self:
        self._df = self._df.drop(columns=column)
        return self

    def drop_duplicates(self, keep: Keep) -> Self:
        self._df = self._df.drop_duplicates(keep=keep, subset=CANONICAL_ID)
        return self


@final
class PolarsDF(Frame[pl.DataFrame, pl.Series], CanonicalIdMixin):
    """Polars DataFrame wrapper"""

    def __init__(self, df: pl.DataFrame, id: str | None = None):
        self._df: pl.DataFrame = self._add_canonical_id(df, id)
        self._id = id

    def _df_as_is(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def _df_overwrite_id(self, df: pl.DataFrame, id: str) -> pl.DataFrame:
        return df.with_columns(df[id].alias(CANONICAL_ID))

    def _df_copy_id(self, df: pl.DataFrame, id: str) -> pl.DataFrame:
        return df.with_columns(df[id].alias(CANONICAL_ID))

    def _df_autoincrement_id(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.arange(0, len(df)).alias(CANONICAL_ID))

    # WRAPPER METHODS:

    @staticmethod
    def _fill_na(series: pl.Series, value: str) -> pl.Series:
        return series.fill_null(value)

    def _get_col(self, column: str) -> pl.Series:
        return self._df.get_column(column)

    def _get_cols(self, columns: tuple[str, ...]) -> pl.DataFrame:
        return self._df.select(columns)

    def put_col(self, column: str, array) -> Self:
        array = pl.Series(array)  # important; allow list to be assigned to column
        self._df = self._df.with_columns(**{column: array})
        return self

    def drop_col(self, column: str) -> Self:
        self._df = self._df.drop(column)
        return self

    def drop_duplicates(self, keep: Keep) -> Self:
        self._df = self._df.unique(keep=keep, subset=CANONICAL_ID, maintain_order=True)
        return self


SparkObject: TypeAlias = spark.DataFrame | RDD[Row]


@final
class SparkDF(Frame[SparkObject, None], CanonicalIdMixin):
    """Spark DataFrame and Spark RDD wrapper

    This wrapper, contrarily to others does not always add a canonical id. When
    canonical ids are to be added the DataFrame is converted to an RDD for
    downstream processing in Worker nodes.

    The `is_init` flag is then used, when False, to keep a high-level
    DataFrame such that it is easier to drop the canonical_id if within
    `drop_duplicates` regime. Also, then the DataFrame is ready for unwrapping
    and feeding back to the user.

    Note:
        Spark DataFrames have to be converted to RDDs as that is the only way
        to create an autoincrementing field.

    Args:
        df: the dataframe
        id: the label of any other id columns used for creation of canonical_id
        is_init: define whether to route the DataFrame to an RDD along with
            canonical ID creation, or not.
    """

    err_msg = "Method is available for spark RDD, not spark DataFrame"

    def __init__(
        self,
        df: spark.DataFrame,
        id: str | None = None,
        is_init: bool = True,
    ):
        # new spark plan for safety
        df = df.select("*")

        self._df: SparkObject
        if is_init:
            self._df: RDD[Row] = self._add_canonical_id(df, id)
        else:
            self._df: spark.DataFrame = df

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
        """Recreate the schema of the dataframe dynamically based on the type
        of the id field.
        """
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

        Permits the access of the base Dedupe class attribute dataframe to be
        returned as a DataFrame even if no canonicalisation has been applied
        yet. For example this would be needed if inspecting the dataframe as
        contained in an instance of Dedupe having yet to call the canonicalizer
        on the set of strategies"""
        if isinstance(self._df, RDD):
            return self._df.toDF()
        return self._df

    # WRAPPER METHODS:

    def drop_col(self, column: str) -> Self:
        """Only applies to Spark DataFrame to remove canonical ID"""
        self._df = self._df.drop(column)
        return self

    def put_col(self):
        raise NotImplementedError(self.err_msg)

    def _get_col(self):
        raise NotImplementedError(self.err_msg)

    def _get_cols(self):
        raise NotImplementedError(self.err_msg)

    def drop_duplicates(self):
        raise NotImplementedError(self.err_msg)


@final
class SparkRows(Frame[list[spark.Row], list[Any]]):
    """Spark Rows DataFrame

    Spark Rows is what are processed by individual Worker nodes.

    Thus, the `Dedupe` entrypoint is able to process a Spark Rows as `Dedupe`
    will be instantiated in the worker node.
    """

    def __init__(self, df: list[spark.Row]):
        self._df: list[spark.Row] = df

    # WRAPPER METHODS:

    @staticmethod
    def _fill_na(series: list, value: str) -> list:
        return [value if v is None else v for v in series]

    def _get_col(self, column: str) -> list[Any]:
        return [row[column] for row in self._df]

    def _get_cols(self, columns: tuple[str, ...]) -> list[spark.Row]:
        return [[row[c] for c in columns] for row in self._df]

    def put_col(self, column: str, array) -> Self:
        array = [i.item() if isinstance(i, np.generic) else i for i in array]
        self._df = [spark.Row(**{**row.asDict(), column: value}) for row, value in zip(self._df, array)]
        return self

    def drop_duplicates(self, keep: Keep) -> Self:

        seen: set[Hashable] = set()
        result: list[Row] = []

        iterable = self._df if keep == "first" else reversed(self._df)

        for row in iterable:
            key = row[CANONICAL_ID]
            if key not in seen:
                seen.add(key)
                result.append(row)

        if keep == "last":
            result.reverse()

        self._df = result
        return self


# DISPATCHER:


@singledispatch
def wrap(df: DataFrameLike, id: str | None = None):
    """
    Wrap the dataframe with instance of `Frame`, for a generic interface
    allowing use of selected methods such as "dropping columns",
    "filling nulls" etc.
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
