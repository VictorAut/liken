"""Deduplication collectionexecutors.

`SparkExecutor` simply calls a partition processor where each partition will
then be processed with the `LocalExecutor`
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import final

import pandas as pd

from liken._collections import DeduplicationDict
from liken._collections import Pipeline
from liken._types import Columns
from liken._types import Keep
from liken.core.executor import Executor


if TYPE_CHECKING:
    from liken.backends.ray.wrapper import RayDF


@final
class RayExecutor(Executor):
    def __init__(self):
        import ray

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    def execute(
        self,
        df: RayDF,
        /,
        *,
        columns: Columns | None,
        dedupers: DeduplicationDict | Pipeline,
        keep: Keep,
        drop_duplicates: bool,
        drop_canonical_id: bool,
        id: str | None = None,
    ) -> RayDF:
        """Maps dataframe partitions to be processed by Ray enforced to use
        pandas for each partition.
        """

        from liken.backends.ray.wrapper import RayDF
        from liken.liken import Dedupe

        def _process_batch(batch: pd.DataFrame) -> pd.DataFrame:
            return (
                Dedupe(batch)  # type: ignore
                .apply(dedupers)  # type: ignore
                .canonicalize(
                    columns,
                    keep=keep,
                    drop_duplicates=drop_duplicates,
                    id=id,
                )
                .collect()
            )

        # IMPORTANT: "pandas" batch
        df = RayDF(df._df.map_batches(_process_batch, batch_format="pandas"))  # type: ignore

        if drop_canonical_id:
            return df.drop_col("canonical_id")
        return df
