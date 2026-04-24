"""Deduplication collectionexecutors.

`PysparkExecutor` simply calls a partition processor where each partition will
then be processed with the `LocalExecutor`
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING
from typing import Type
from typing import final

from liken.constants import CANONICAL_ID
from liken.core.executor import Executor


if TYPE_CHECKING:
    from pyspark.sql import Row
    from pyspark.sql import SparkSession

    from liken.backends.pyspark.wrapper import PysparkDF
    from liken.collections.base import Pipeline
    from liken.collections.dict import DeduplicationDict
    from liken.liken import Dedupe
    from liken.types import Columns
    from liken.types import Keep


@final
class PysparkExecutor(Executor):
    def __init__(self, spark_session: SparkSession):
        self._spark_session = spark_session

    def execute(
        self,
        df: PysparkDF,
        /,
        *,
        columns: Columns | None,
        dedupers: DeduplicationDict | Pipeline,
        keep: Keep,
        drop_duplicates: bool,
        drop_canonical_id: bool,
        id: str | None = None,
    ) -> PysparkDF:
        """Maps dataframe partitions to be processed via the RDD API yielding
        low-level list[Rows], which are then post-processed back to a dataframe.
        """

        # import in worker node
        from liken.backends.pyspark.wrapper import PysparkDF
        from liken.liken import Dedupe

        # IMPORTANT: Use local variables, no references to Self
        # Allows for serialization via Py4J
        process_partition = self._process_partition

        rdd = df.mapPartitions(
            lambda partition: process_partition(
                factory=Dedupe,
                partition=partition,
                dedupers=dedupers,
                id=id,
                columns=columns,
                drop_duplicates=drop_duplicates,
                keep=keep,
            )
        )

        schema = df._schema

        df = PysparkDF(self._spark_session.createDataFrame(rdd, schema=schema), is_init=False)

        if drop_canonical_id:
            return df.drop_col(CANONICAL_ID)
        return df

    @staticmethod
    def _process_partition(
        *,
        factory: Type[Dedupe],
        partition: Iterator[Row],
        dedupers: DeduplicationDict | Pipeline,
        id: str | None,
        columns: Columns | None,
        drop_duplicates: bool,
        keep: Keep = "first",
    ) -> Iterator[Row]:
        """process a spark dataframe partition i.e. a list[Row]

        This function is functionality mapped to a worker node. For clean
        separation from the driver, dedupers are re-instantiated and the main
        liken API is executed *per* worker node.

        Args:
            paritition_iter: a partition
            dedupers: the collection of dedupers
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
        df = (
            factory._from_rows(rows)
            .apply(dedupers)  # type: ignore
            .canonicalize(
                columns,
                keep=keep,
                drop_duplicates=drop_duplicates,
                id=id,
            )
            .collect()
        )

        return iter(df)  # type: ignore
