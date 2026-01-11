from __future__ import annotations
from collections.abc import Iterator
from typing import final, Protocol, TYPE_CHECKING, TypeVar

from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructField, StructType

from dupegrouper.constants import (
    CANONICAL_ID,
    PYSPARK_TYPES,
)
from dupegrouper.dataframe import DF, LocalDF, SparkDF
from dupegrouper.strats_library import BaseStrategy
from dupegrouper.strats_manager import DEFAULT_STRAT_KEY, StratsConfig
from dupegrouper.types import Columns, Keep

if TYPE_CHECKING:
    from dupegrouper.base import Duped
    

class Executor(Protocol[DF]):
    def canonicalize(
        self,
        df: DF,
        columns: Columns | None,
        strats: StratsConfig,
    ) -> DF: ...


@final
class LocalExecutor(Executor):

    def __init__(self, keep: Keep):
        self._keep = keep

    def canonicalize(
        self,
        df: LocalDF,
        columns: Columns | None,
        strats: StratsConfig,
    ) -> LocalDF:
        if not columns:
            for col, iter_strats in strats.items():
                for strat in iter_strats:
                    df = self._call_strat(strat, df, col)
            return df

        # For inline calls e.g.`.canonicalize("address")`
        for strat in strats[DEFAULT_STRAT_KEY]:
            df = self._call_strat(strat, df, columns)
        return df

    def _call_strat(
        self,
        strat: BaseStrategy,
        df: LocalDF,
        columns: Columns,
    ) -> LocalDF:
        return (
            strat
            #
            .set_frame(df)
            .set_keep(self._keep)
            .canonicalize(columns)
        )


@final
class SparkExecutor(Executor):

    def __init__(self, keep: Keep, spark_session: SparkSession, id):
        self._keep = keep
        self._spark_session = spark_session
        self._id = id

    def canonicalize(
        self,
        df: SparkDF,
        columns: Columns | None,
        strats: StratsConfig,
    ) -> SparkDF:
        """Spark specific deduplication helper

        Maps dataframe partitions to be processed via the RDD API yielding low-
        level list[Rows], which are then post-processed back to a dataframe.

        Args:
            columns: The attribute to deduplicate.
            strats: the collection of strats
        Retuns:
            Instance's _df attribute is updated
        """

        from dupegrouper.base import Duped

        # IMPORTANT: Use local variables, not references to `self`
        id = self._id
        keep = self._keep
        process_partition = self._process_partition

        # IMPORTANT: Do not pass any references to `self`
        rdd = df.rdd.mapPartitions(
            lambda partition: process_partition(
                factory=Duped,
                partition=partition,
                strats=strats,
                id=id,
                columns=columns,
                keep=keep,
            )
        )

        schema = self._get_schema(df)

        df = SparkDF(
            self._spark_session.createDataFrame(rdd, schema=schema),
            id,
        )
        return df

    def _get_schema(self, df: SparkDF) -> StructType:
        fields = df.schema.fields
        if CANONICAL_ID not in df.columns:
            id_type = PYSPARK_TYPES.get(dict(df.dtypes).get(self._id))
            fields += [StructField(CANONICAL_ID, id_type, True)]
        return StructType(fields)

    @staticmethod
    def _process_partition(
        factory: Duped,
        partition: Iterator[Row],
        strats: StratsConfig,
        id: str,
        columns: Columns | None,
        keep: Keep = "first",
    ) -> Iterator[Row]:
        """process a spark dataframe partition i.e. a list[Row]

        This function is functionality mapped to a worker node. For clean
        separation from the driver, strats are re-instantiated and the main
        dupegrouper API is executed *per* worker node.

        Args:
            paritition_iter: a partition
            strats: the collection of strats
            id: the unique identified of the dataset a.k.a "business key"
            columns: the attribute on which to deduplicate

        Returns:
            A list[Row], deduplicated
        """
        # handle empty partitions
        rows: list[Row] = list(partition)
        if not rows:
            return iter([])

        # Core API reused per partition, per worker node
        dp = factory(rows, id=id, keep=keep)
        dp.apply(strats)
        dp.canonicalize(columns)

        return iter(dp.df)
