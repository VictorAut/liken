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

from __future__ import annotations

from collections.abc import Hashable
from functools import singledispatch
from typing import Any
from typing import Generic
from typing import Protocol
from typing import Self
from typing import TypeAlias
from typing import TypeVar
from typing import final

import dask.dataframe as dd
import modin.pandas as mpd
import pandas as pd
import polars as pl
import pyarrow as pa
import pyspark.sql as spark
import ray
from pyarrow.compute import coalesce
from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql import functions
from pyspark.sql.types import LongType
from pyspark.sql.types import StructField
from pyspark.sql.types import StructType
from ray.data import Dataset as RayDataset
from typing_extensions import override

from liken._constants import CANONICAL_ID
from liken._constants import NA_PLACEHOLDER
from liken._constants import PYSPARK_TYPES
from liken._types import Columns
from liken._types import Keep


# TYPES


D = TypeVar("D")  # dataframe


# BASE


class Frame(Generic[D]):
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

    def _get_col(self, column: str) -> pa.Array:
        del column
        raise NotImplementedError

    def _get_cols(self, columns: tuple[str, ...]) -> pa.Table:
        del columns
        raise NotImplementedError

    def get_array(self, columns: Columns, with_na: bool = False) -> pa.Array | pa.Table:
        """Generalise the getting of a column, or columns of a df to an array.

        Handles single column and multicolumn. For instances of single column
        the initial column can initially be filled null placeholders, to allow
        for use by dedupers. This is optional so that specific dedupers
        that do care about nulls are not affected (e.g. IsNA).
        """
        if isinstance(columns, str):
            col: pa.Array = self._get_col(columns)
            if with_na:
                return coalesce(col, NA_PLACEHOLDER)
            return col
        return self._get_cols(columns)

    def get_canonical(self) -> pa.Array:
        """Convenience method"""
        return self.get_array(CANONICAL_ID)

    def synthesize_record(self) -> D:
        raise NotImplementedError


# CANONICAL ID


class AddsCanonical(Protocol):
    """Mixin protocol"""

    def _df_as_is(self, df): ...
    def _df_overwrite_id(self, df, id: str): ...
    def _df_copy_id(self, df, id: str): ...
    def _df_autoincrement_id(self, df): ...
    def _column_labels_list(self, df): ...


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

        has_canonical: bool = CANONICAL_ID in self._column_labels_list(df)
        id_is_canonical: bool = id == CANONICAL_ID

        if has_canonical:
            if id:
                if id_is_canonical:
                    return self._df_as_is(df)
                # overwrite with id
                return self._df_overwrite_id(df, id)
            return self._df_as_is(df)
        if id:
            # write new with id
            return self._df_copy_id(df, id)
        # write new auto-incrementing
        return self._df_autoincrement_id(df)


# WRAPPERS


@final
class PandasDF(Frame[pd.DataFrame], CanonicalIdMixin):
    """Pandas DataFrame wrapper"""

    def __init__(self, df: pd.DataFrame, id: str | None = None):
        self._df: pd.DataFrame = self._add_canonical_id(df, id)
        self._id = id

    # CANONICAL ID HELPERS:

    def _df_as_is(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _df_overwrite_id(self, df: pd.DataFrame, id: str) -> pd.DataFrame:
        return df.assign(**{CANONICAL_ID: df[id]})

    def _df_copy_id(self, df: pd.DataFrame, id: str) -> pd.DataFrame:
        return df.assign(**{CANONICAL_ID: df[id]})

    def _df_autoincrement_id(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(**{CANONICAL_ID: pd.RangeIndex(start=0, stop=len(df))})

    def _column_labels_list(self, df: pd.DataFrame) -> list[str]:
        return list(df.columns)

    # ARROW INTERFACES:

    def _get_col(self, column: str) -> pa.Array:
        return pa.array(self._df[column])

    def _get_cols(self, columns: tuple[str, ...]) -> pa.Table:
        return pa.Table.from_pandas(self._df[list(columns)])

    # WRAPPER METHODS:

    def put_col(self, column: str, array: list) -> Self:
        self._df = self._df.assign(**{column: array})
        return self

    def drop_col(self, column: str) -> Self:
        self._df = self._df.drop(columns=column)
        return self

    def drop_duplicates(self, keep: Keep) -> Self:
        self._df = self._df.drop_duplicates(keep=keep, subset=CANONICAL_ID)
        return self

    # SYNTHETIC RECORD:

    def synthesize_record(self) -> pd.DataFrame:
        def _first_non_null(series: pd.Series):
            non_null = series.dropna()
            return non_null.iloc[0] if not non_null.empty else pd.NA

        return self._df.groupby(CANONICAL_ID, as_index=False).agg(_first_non_null)


@final
class PolarsDF(Frame[pl.DataFrame], CanonicalIdMixin):
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


SparkObject: TypeAlias = spark.DataFrame | RDD[Row]


@final
class SparkDF(Frame[SparkObject], CanonicalIdMixin):
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

    err_msg = "Method is available for spark Rows only, not spark DataFrame"

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
            self._df: RDD[Row] = self._add_canonical_id(df, id)  # type: ignore[no-redef]
        else:
            self._df: spark.DataFrame = df  # type: ignore[no-redef]

        self._id = id

    # CANONICAL ID HELPERS:

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

    def _column_labels_list(self, df: spark.DataFrame) -> list[str]:
        return df.columns

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
        on the collection of dedupers."""
        if isinstance(self._df, RDD):
            return self._df.toDF()
        return self._df

    # WRAPPER METHODS:

    def drop_col(self, column: str) -> Self:
        """Only applies to Spark DataFrame to remove canonical ID"""
        if isinstance(self._df, spark.DataFrame):
            self._df = self._df.drop(column)
            return self
        raise NotImplementedError("Cannot drop columns on spark RDD")

    def put_col(self):
        raise NotImplementedError(self.err_msg)

    def _get_cols(self):
        raise NotImplementedError(self.err_msg)

    def _get_col(self, column: str):
        """Everything is returned to driver node!"""
        df: spark.DataFrame = self._df.toDF()

        return df.select(column).toArrow().column(0).combine_chunks()

    def drop_duplicates(self):
        raise NotImplementedError(self.err_msg)

    # SYNTHETIC RECORD:

    def synthesize_record(self) -> spark.DataFrame:
        """TODO
        warn that everything is returned to driver node!
        """
        df: spark.DataFrame = self._df.toDF()
        exprs = []

        for col in df.columns:
            if col == CANONICAL_ID:
                continue

            exprs.append(functions.first(functions.col(col), ignorenulls=True).alias(col))

        return df.groupBy(CANONICAL_ID).agg(*exprs).orderBy(CANONICAL_ID)


@final
class SparkRows(Frame[list[spark.Row]]):
    """Spark Rows DataFrame

    Spark Rows is what are processed by individual Worker nodes.

    Thus, the `Dedupe` entrypoint is able to process a Spark Rows as `Dedupe`
    will be instantiated in the worker node.
    """

    def __init__(self, df: list[spark.Row]):
        self._df: list[spark.Row] = df

    # ARROW INTERFACES:

    def _get_col(self, column: str) -> pa.Array:
        return pa.array([row[column] for row in self._df])

    def _get_cols(self, columns: tuple[str, ...]) -> pa.Table:

        data = {col: [row[col] for row in self._df] for col in columns}

        return pa.table(data)

    # WRAPPER METHODS:

    def put_col(self, column: str, array: list) -> Self:
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


@final
class ModinDF(Frame[mpd.DataFrame], CanonicalIdMixin):
    """Modin DataFrame wrapper"""

    def __init__(self, df: mpd.DataFrame, id: str | None = None):
        self._df: mpd.DataFrame = self._add_canonical_id(df, id)
        self._id = id

    # CANONICAL ID HELPERS:

    def _df_as_is(self, df: mpd.DataFrame) -> mpd.DataFrame:
        return df

    def _df_overwrite_id(self, df: mpd.DataFrame, id: str) -> mpd.DataFrame:
        return df.assign(**{CANONICAL_ID: df[id]})

    def _df_copy_id(self, df: mpd.DataFrame, id: str) -> mpd.DataFrame:
        return df.assign(**{CANONICAL_ID: df[id]})

    def _df_autoincrement_id(self, df: mpd.DataFrame) -> mpd.DataFrame:
        return df.assign(**{CANONICAL_ID: mpd.RangeIndex(start=0, stop=len(df))})

    def _column_labels_list(self, df: mpd.DataFrame) -> list[str]:
        return df.columns

    # ARROW INTERFACES:

    def _get_col(self, column: str) -> pa.Array:
        return pa.array(self._df[column]._to_pandas())

    def _get_cols(self, columns: tuple[str, ...]) -> pa.Table:
        return pa.Table.from_pandas(
            self._df[list(columns)]._to_pandas()
        )  # TODO: there's a to_pandas() public function?

    # WRAPPER METHODS:

    def put_col(self, column: str, array: list) -> Self:
        self._df = self._df.assign(**{column: array})
        return self

    def drop_col(self, column: str) -> Self:
        self._df = self._df.drop(columns=column)
        return self

    def drop_duplicates(self, keep: Keep) -> Self:
        self._df = self._df.drop_duplicates(keep=keep, subset=CANONICAL_ID)
        return self

    # SYNTHETIC RECORD:

    def synthesize_record(self) -> mpd.DataFrame:
        def _first_non_null(series):
            non_null = series.dropna()
            return non_null.iloc[0] if not non_null.empty else None

        return self._df.groupby(CANONICAL_ID, as_index=False).agg(_first_non_null)


@final
class RayDF(Frame[RayDataset], CanonicalIdMixin):
    """Ray Dataset wrapper"""

    def __init__(self, df: RayDataset, id: str | None = None):
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        self._df: RayDataset = self._add_canonical_id(df, id)
        self._id = id

    # CANONICAL ID HELPERS:

    def _df_as_is(self, df: RayDataset) -> RayDataset:
        return df

    def _df_overwrite_id(self, df: RayDataset, id: str) -> RayDataset:
        return df.map_batches(lambda df: df.assign(**{CANONICAL_ID: df[id]}), batch_format="pandas")

    def _df_copy_id(self, df: RayDataset, id: str) -> RayDataset:
        return df.map_batches(lambda df: df.assign(**{CANONICAL_ID: df[id]}), batch_format="pandas")

    def _df_autoincrement_id(self, df):
        """collects data to driver node!"""
        # Get block sizes
        block_sizes = df.map_batches(lambda df: pd.DataFrame({"__len__": [len(df)]}), batch_format="pandas").take_all()

        lengths = [row["__len__"] for row in block_sizes]

        # Compute offsets
        offsets = []
        total = 0
        for length in lengths:
            offsets.append(total)
            total += length

        # Assign offsets per block
        def _add_id_generator():
            for i, block in enumerate(df.iter_batches(batch_format="pandas")):
                offset = offsets[i]
                block = block.reset_index(drop=True)
                block[CANONICAL_ID] = range(offset, offset + len(block))
                yield block

        return ray.data.from_pandas_refs([ray.put(batch) for batch in _add_id_generator()])

    def _column_labels_list(self, df: RayDataset) -> list[str]:
        return df.columns()

    # ARROW INTERFACES:

    def _get_col(self, column: str) -> pa.Array:
        # TODO: in a future Ray version, type check
        return self._df.to_arrow()[column]  # type: ignore[attr-defined]

    def _get_cols(self, columns: tuple[str, ...]) -> pa.Table:
        # TODO: in a future Ray version, type check
        return self._df.to_arrow().select(columns)  # type: ignore[attr-defined]

    # WRAPPER METHODS:

    def put_col(self, column: str, array: list) -> Self:
        def fn(df):
            df[column] = array
            return df

        self._df = self._df.map_batches(fn)
        return self

    def drop_col(self, column: str) -> Self:
        self._df = self._df.drop_columns([column])
        return self

    def drop_duplicates(self, keep: Keep) -> Self:
        def fn(df):
            return df.drop_duplicates(subset=[CANONICAL_ID], keep=keep)

        self._df = self._df.map_batches(fn)
        return self

    # SYNTHETIC RECORD:

    def synthesize_record(self):

        def fn(df: pd.DataFrame):
            return df.groupby(CANONICAL_ID, as_index=False).first()

        return self._df.map_batches(fn)


@final
class DaskDF(Frame[dd.DataFrame], CanonicalIdMixin):
    """Dask DataFrame wrapper"""

    def __init__(self, df: dd.DataFrame, id: str | None = None, preserve_schema: bool = False):
        if preserve_schema:
            self._df = df
        else:
            self._df = self._add_canonical_id(df, id)
        self._id = id

    # CANONICAL ID HELPERS:

    def _df_as_is(self, df: dd.DataFrame) -> dd.DataFrame:
        return df

    def _df_overwrite_id(self, df: dd.DataFrame, id: str) -> dd.DataFrame:
        return df.assign(**{CANONICAL_ID: df[id]})

    def _df_copy_id(self, df: dd.DataFrame, id: str) -> dd.DataFrame:
        return df.assign(**{CANONICAL_ID: df[id]})

    def _df_autoincrement_id(self, df: dd.DataFrame) -> dd.DataFrame:

        lengths = df.map_partitions(len).compute()
        offsets = []
        total = 0
        for length in lengths:
            offsets.append(total)
            total += length

        def _assign(partition, partition_info=None):
            i = partition_info["number"]
            offset = offsets[i]
            partition = partition.reset_index(drop=True)
            partition[CANONICAL_ID] = range(offset, offset + len(partition))
            return partition

        meta = self._new_meta(df)

        return df.map_partitions(_assign, meta=meta)

    def _new_meta(self, df: dd.DataFrame, id: str | None = None) -> pd.DataFrame:
        meta = df._meta.copy()

        dtype = meta[id].dtype if id else "int64"

        meta[CANONICAL_ID] = pd.Series(dtype=dtype)

        return meta

    def _column_labels_list(self, df: dd.DataFrame) -> list[str]:
        return df.columns

    # ARROW INTERFACES:

    def _get_col(self, column: str) -> pa.Array:
        return pa.array(self._df[column].compute())

    def _get_cols(self, columns: tuple[str, ...]) -> pa.Table:
        return pa.Table.from_pandas(self._df[list(columns)].compute())

    # WRAPPER METHODS:

    def put_col(self, column: str, array: list) -> Self:
        self._df = self._df.assign(**{column: array})
        return self

    def drop_col(self, column: str) -> Self:
        self._df = self._df.drop(columns=column)
        return self

    def drop_duplicates(self, keep: Keep) -> Self:
        self._df = self._df.drop_duplicates(subset=CANONICAL_ID, keep=keep)
        return self

    # SYNTHETIC RECORD:

    def synthesize_record(self) -> dd.DataFrame:
        def _first_non_null(series):
            non_null = series.dropna()
            return non_null.iloc[0] if len(non_null) else None

        return self._df.groupby(CANONICAL_ID).agg(_first_non_null).reset_index()


# DISPATCHER:


@singledispatch
def wrap(
    df: pd.DataFrame | pl.DataFrame | mpd.DataFrame | spark.DataFrame | list[spark.Row],
    id: str | None = None,
):
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


@wrap.register(mpd.DataFrame)
def _(df, id: str | None = None) -> ModinDF:
    return ModinDF(df, id)


@wrap.register(RayDataset)
def _(df, id: str | None = None) -> RayDF:
    return RayDF(df, id)


@wrap.register(dd.DataFrame)
def _(df, id: str | None = None) -> DaskDF:
    return DaskDF(df, id)


@wrap.register(spark.DataFrame)
def _(df, id: str | None = None) -> SparkDF:
    return SparkDF(df, id)


@wrap.register(list)
def _(df: list[spark.Row], id: str | None) -> SparkRows:
    del id
    return SparkRows(df)


# ACCESSIBLE TYPES


LocalDF: TypeAlias = PandasDF | PolarsDF | ModinDF | SparkRows
DistributedDF: TypeAlias = SparkDF | RayDF | DaskDF
DF = TypeVar("DF", LocalDF, DistributedDF)
