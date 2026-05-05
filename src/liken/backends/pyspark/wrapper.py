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
Spark DataFrame, the wrapper will call the PysparkDF class which will create
canonical IDs. However the output to this is RDDs which are then processed
by the executor into Spark Rows which are dispatched to worker nodes. Spark
Rows can be fully recovered to a Spark DataFrame using the same PysparkDF class.

TODO:
    - CanonicalIdMixin should be defined first when inherited
    - A full interface can then be defined
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING
from typing import Self
from typing import TypeAlias
from typing import final

import pyarrow as pa
from typing_extensions import override

from liken.constants import CANONICAL_ID
from liken.core.wrapper import DF
from liken.core.wrapper import CanonicalIdMixin


if TYPE_CHECKING:
    from pyspark.rdd import RDD
    from pyspark.sql import DataFrame
    from pyspark.sql import Row
    from pyspark.sql.types import StructType

    from liken.types import Keep

    SparkObject: TypeAlias = DataFrame | RDD[Row]


@final
class PysparkDF(DF["SparkObject"], CanonicalIdMixin):
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
        df: DataFrame,
        id: str | None = None,
        is_init: bool = True,
    ):
        # new spark plan for safety
        df = df.select("*")

        self._df: SparkObject
        if is_init:
            self._df: RDD[Row] = self._add_canonical_id(df, id)  # type: ignore[no-redef]
        else:
            self._df: DataFrame = df  # type: ignore[no-redef]

        self._id = id

    # CANONICAL ID HELPERS:

    def _df_as_is(self, df: DataFrame) -> RDD[Row]:
        self._schema = df.schema
        return df.rdd

    def _df_overwrite_id(self, df: DataFrame, id: str) -> RDD[Row]:

        from pyspark.sql import Row

        df_new: DataFrame = df.drop(CANONICAL_ID)
        self._schema = self._new_schema(df_new, id)
        return df_new.rdd.mapPartitions(
            lambda partition: [Row(**{**row.asDict(), CANONICAL_ID: row[id]}) for row in partition]
        )

    def _df_copy_id(self, df: DataFrame, id: str) -> RDD[Row]:

        from pyspark.sql import Row

        self._schema = self._new_schema(df, id)
        return df.rdd.mapPartitions(
            lambda partition: [Row(**{**row.asDict(), CANONICAL_ID: row[id]}) for row in partition]
        )

    def _df_autoincrement_id(self, df: DataFrame) -> RDD[Row]:

        from pyspark.sql import Row

        self._schema = self._new_schema(df)
        return df.rdd.zipWithIndex().mapPartitions(
            lambda partition: [Row(**{**row.asDict(), CANONICAL_ID: idx}) for row, idx in partition]
        )

    def _column_labels_list(self, df: DataFrame) -> list[str]:
        return df.columns

    @staticmethod
    def _new_schema(df: DataFrame, id: str | None = None) -> StructType:
        """Recreate the schema of the dataframe dynamically based on the type
        of the id field.
        """
        # lazy
        import pyspark.sql.types as T

        pyspark_types: dict = {
            "boolean": T.BooleanType(),
            "date": T.DateType(),
            "double": T.DoubleType(),
            "float": T.FloatType(),
            "int": T.IntegerType(),
            "bigint": T.LongType(),
            "string": T.StringType(),
            "timestamp": T.TimestampType(),
        }

        fields = df.schema.fields
        if id:
            dtype = dict(df.dtypes)[id]
            id_type = pyspark_types[dtype]
        else:
            id_type = T.LongType()  # auto-incremental is numeric
        fields += [T.StructField(CANONICAL_ID, id_type, True)]
        return T.StructType(fields)

    @override
    def unwrap(self) -> DataFrame:
        """Ensure the unwrapped dataframe is always an instance of DataFrame

        Permits the access of the base Dedupe class attribute dataframe to be
        returned as a DataFrame even if no canonicalisation has been applied
        yet. For example this would be needed if inspecting the dataframe as
        contained in an instance of Dedupe having yet to call the canonicalizer
        on the collection of dedupers."""
        from pyspark.rdd import RDD

        if isinstance(self._df, RDD):
            return self._df.toDF()
        return self._df

    # WRAPPER METHODS:

    def drop_col(self, column: str) -> Self:
        """Only applies to Spark DataFrame to remove canonical ID"""
        from pyspark.sql import DataFrame

        if isinstance(self._df, DataFrame):
            self._df = self._df.drop(column)
            return self
        raise NotImplementedError("Cannot drop columns on spark RDD")

    def put_col(self):
        raise NotImplementedError(self.err_msg)

    def _get_cols(self):
        raise NotImplementedError(self.err_msg)

    def _get_col(self, column: str):
        """Everything is returned to driver node!"""
        df: DataFrame = self._df.toDF()

        return df.select(column).toArrow().column(0).combine_chunks()

    def drop_duplicates(self):
        raise NotImplementedError(self.err_msg)

    # SYNTHETIC RECORD:

    def synthesize_record(self) -> DataFrame:
        """TODO
        warn that everything is returned to driver node!
        """
        # lazy
        from pyspark.sql import functions

        df: DataFrame = self._df.toDF()
        exprs = []

        for col in df.columns:
            if col == CANONICAL_ID:
                continue

            exprs.append(functions.first(functions.col(col), ignorenulls=True).alias(col))

        return df.groupBy(CANONICAL_ID).agg(*exprs).orderBy(CANONICAL_ID)


@final
class PysparkRows(DF["list[Row]"]):
    """Spark Rows DataFrame

    Spark Rows is what are processed by individual Worker nodes.

    Thus, the `Dedupe` entrypoint is able to process a Spark Rows as `Dedupe`
    will be instantiated in the worker node.
    """

    def __init__(self, df: list[Row]):
        self._df: list[Row] = df

    # ARROW INTERFACES:

    def _get_col(self, column: str) -> pa.Array:
        return pa.array([row[column] for row in self._df])

    def _get_cols(self, columns: tuple[str, ...]) -> pa.Table:

        data = {col: [row[col] for row in self._df] for col in columns}

        return pa.table(data)

    # WRAPPER METHODS:

    def put_col(self, column: str, array: list) -> Self:
        from pyspark.sql import Row

        self._df = [Row(**{**row.asDict(), column: value}) for row, value in zip(self._df, array)]
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
