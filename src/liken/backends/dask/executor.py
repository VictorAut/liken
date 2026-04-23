"""Deduplication collectionexecutors.

`SparkExecutor` simply calls a partition processor where each partition will
then be processed with the `LocalExecutor`
"""

from __future__ import annotations

from typing import final

import pandas as pd

from liken._collections import DeduplicationDict
from liken._collections import Pipeline
from liken._constants import CANONICAL_ID
from liken._types import Columns
from liken._types import Keep

# if TYPE_CHECKING:
from liken.backends.dask.wrapper import DaskDF
from liken.core.executor import Executor


@final
class DaskExecutor(Executor):
    def execute(
        self,
        df: DaskDF,
        /,
        *,
        columns: Columns | None,
        dedupers: DeduplicationDict | Pipeline,
        keep: Keep,
        drop_duplicates: bool,
        drop_canonical_id: bool,
        id: str | None = None,
    ) -> DaskDF:
        """Maps dataframe partitions to be processed by Dask, natively using
        pandas for each partition.
        """

        meta = df._new_meta(df._df, id)

        if drop_canonical_id:
            meta = meta.drop(columns=[CANONICAL_ID])

        process_partition = self._process_partition

        df = DaskDF(
            df._df.map_partitions(
                process_partition,
                dedupers=dedupers,
                columns=columns,
                keep=keep,
                drop_duplicates=drop_duplicates,
                drop_canonical_id=drop_canonical_id,
                id=id,
                meta=meta,
            ),
            id=id,
            preserve_schema=True,
        )

        return df

    @staticmethod
    def _process_partition(
        df: pd.DataFrame,
        dedupers: DeduplicationDict | Pipeline,
        columns: Columns | None,
        keep: Keep,
        drop_duplicates: bool,
        drop_canonical_id: bool,
        id: str | None = None,
    ) -> pd.DataFrame:
        from liken.liken import Dedupe

        df = (
            Dedupe(df)  # type: ignore
            .apply(dedupers)  # type: ignore
            .canonicalize(
                columns,
                keep=keep,
                drop_duplicates=drop_duplicates,
                id=id,
            )
            .collect()
        )

        if drop_canonical_id:
            return df.drop(columns=[CANONICAL_ID])
        return df
