from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import cast, final

from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructField, StructType, DataType

from dupegrouper.constants import (
    CANONICAL_ID,
    DEFAULT_STRAT_KEY,
    PYSPARK_TYPES,
)
from dupegrouper.dataframe import WrappedSparkDataFrame
from dupegrouper.strats_manager import StratsConfig
from dupegrouper.types import Columns, Rule


class Executor(ABC):

    @abstractmethod
    def canonicalize(
        self,
        columns: Columns | None,
        strategies: StratsConfig,
    ) -> None:
        pass


@final
class LocalExecutor(Executor):

    def __init__(self, canonicalization_rule: Rule):
        self._canonicalization_rule = canonicalization_rule

    def canonicalize(
        self,
        df,
        columns: Columns | None,
        strategies: StratsConfig,
    ) -> None:
        if not columns:
            for col, strategies in strategies.items():
                for strategy in strategies:
                    return (
                        strategy
                        #
                        .set_frame(df)
                        .set_rule(self._canonicalization_rule)
                        .canonicalize(col)
                    )

        # For inline calls e.g.`.canonicalize("address")`
        for strategy in strategies[DEFAULT_STRAT_KEY]:
            return (
                strategy
                #
                .set_frame(df)
                .set_rule(self._canonicalization_rule)
                .canonicalize(columns)
            )


@final
class SparkExecutor(Executor):

    def __init__(self, canonicalization_rule: Rule, spark_session: SparkSession, id):
        self._canonicalization_rule = canonicalization_rule
        self._spark_session = spark_session
        self._id = id

    def canonicalize(
        self,
        df,
        columns: Columns | None,
        strategies: StratsConfig,
    ) -> None:
        """Spark specific deduplication helper

        Maps dataframe partitions to be processed via the RDD API yielding low-
        level list[Rows], which are then post-processed back to a dataframe.

        Args:
            columns: The attribute to deduplicate.
            strategies: the collection of strategies
        Retuns:
            Instance's _df attribute is updated
        """
        id = cast(str, self._id)
        canonicalization_rule = cast(str, self._canonicalization_rule)
        id_type = cast(DataType, PYSPARK_TYPES.get(dict(df.dtypes).get(id)))  # type: ignore

        from dupegrouper.base import Duped

        canonicalized_rdd = df.rdd.mapPartitions(
            lambda partition: _process_partition(
                factory = Duped,
                partition=partition,
                strategies=strategies,
                id=id,
                columns=columns,
                canonicalization_rule=canonicalization_rule,
            )
        )

        if CANONICAL_ID in df.columns:
            schema = StructType(df.schema.fields)
        else:
            schema = StructType(df.schema.fields + [StructField(CANONICAL_ID, id_type, True)])

        df = WrappedSparkDataFrame(
            cast(SparkSession, self._spark_session).createDataFrame(canonicalized_rdd, schema=schema), id
        )
        return df



# This has to be it's own function, can't be a method of SparkExecutor
def _process_partition(
    factory,
    partition: Iterator[Row],
    strategies: StratsConfig,
    id: str,
    columns: str | None,
    canonicalization_rule: Rule = "first",
) -> Iterator[Row]:
    """process a spark dataframe partition i.e. a list[Row]

    This function is functionality mapped to a worker node. For clean
    separation from the driver, strategies are re-instantiated and the main
    dupegrouper API is executed *per* worker node.

    Args:
        paritition_iter: a partition
        strategies: the collection of strategies
        id: the unique identified of the dataset a.k.a "business key"
        columns: the attribute on which to deduplicate

    Returns:
        A list[Row], deduplicated
    """
    # handle empty partitions
    rows = list(partition)
    if not rows:
        return iter([])

    # Core API reused per partition, per worker node
    dp = factory(rows, id=id, canonicalization_rule=canonicalization_rule)
    print(dp.df)
    dp.apply(strategies)
    dp.canonicalize(columns)

    return iter(dp.df)  # type: ignore[arg-type]
