"""Deduplication collection executors.

`PysparkExecutor` simply calls a partition processor where each partition will
then be processed with the `LocalExecutor`
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import final

from liken.core.executor import Executor


if TYPE_CHECKING:
    from collections.abc import Iterator

    from pyspark.sql import Row

    from liken.backends.pyspark.wrapper import PysparkDF
    from liken.collections.base import Pipeline
    from liken.collections.dict import DeduplicationDict
    from liken.liken import Dedupe
    from liken.types import Columns
    from liken.types import Keep


@final
class PysparkExecutor(Executor):
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
        source_id: str | None = None,
        canonical_id: str,
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
                source_id=source_id,
                canonical_id=canonical_id,
                columns=columns,
                drop_duplicates=drop_duplicates,
                keep=keep,
            )
        )

        schema = df._schema

        df = PysparkDF(df.sparkSession.createDataFrame(rdd, schema=schema), canonical_id=canonical_id, is_init=False)

        if drop_canonical_id:
            return df.drop_col(canonical_id)
        return df

    @staticmethod
    def _process_partition(
        *,
        factory: type[Dedupe],
        partition: Iterator[Row],
        dedupers: DeduplicationDict | Pipeline,
        source_id: str | None,
        canonical_id: str,
        columns: Columns | None,
        drop_duplicates: bool,
        keep: Keep = "first",
    ) -> Iterator[Row]:
        """process a spark dataframe partition i.e. a list[Row]

        This function is functionality mapped to a worker node. For clean
        separation from the driver, dedupers are re-instantiated and the main
        liken API is executed *per* worker node.

        Args:
            partition: a partition
            dedupers: the collection of dedupers
            source_id: the unique identifier of the dataset a.k.a "business key"
            columns: the attribute on which to deduplicate

        Returns:
            A list[Row], deduplicated
        """
        # handle empty partitions
        rows: list[Row] = list(partition)
        if not rows:
            return iter([])

        # Core API reused per partition, per worker node
        dupe = factory._from_rows(rows)

        # Internal handoff: the sequential API stores dedupers under the
        # reserved `_default_` key, which a public dict apply must not use.
        # Re-apply them individually to keep the sequential semantics.
        from liken.collections.base import SEQUENTIAL_API_DEFAULT_KEY
        from liken.collections.dict import DeduplicationDict as DictImpl

        if isinstance(dedupers, DictImpl) and set(dedupers) == {SEQUENTIAL_API_DEFAULT_KEY}:
            for deduper in dedupers[SEQUENTIAL_API_DEFAULT_KEY]:
                dupe.apply(deduper)
        else:
            dupe.apply(dedupers)  # type: ignore[arg-type]

        dupe._canonicalize(
            columns=columns,
            keep=keep,
            drop_duplicates=drop_duplicates,
            source_id=source_id,
            canonical_id=canonical_id,
        )
        df = dupe.collect()

        return iter(df)  # type: ignore[arg-type]
