"""Deduplication collection executors.

`PysparkExecutor` simply calls a partition processor where each partition will
then be processed with the `LocalExecutor`
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import final

from liken.constants import CANONICAL_ID
from liken.core.executor import Executor


if TYPE_CHECKING:
    import pandas as pd

    from liken.backends.dask.wrapper import DaskDF
    from liken.collections.base import Pipeline
    from liken.collections.dict import DeduplicationDict
    from liken.types import Columns
    from liken.types import Keep


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

        from liken.backends.dask.wrapper import DaskDF

        meta = df._new_meta(df._df, id)

        if drop_canonical_id:
            meta = meta.drop(columns=[CANONICAL_ID])

        process_partition = self._process_partition

        return DaskDF(
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
        from liken.collections.base import SEQUENTIAL_API_DEFAULT_KEY
        from liken.collections.dict import DeduplicationDict as DictImpl
        from liken.liken import Dedupe

        dupe = Dedupe(df)  # type: ignore[assignment]

        # Internal handoff: the sequential API stores dedupers under the
        # reserved `_default_` key, which a public dict apply must not use
        # (U07). Re-apply them individually to keep the sequential semantics.
        if isinstance(dedupers, DictImpl) and set(dedupers) == {SEQUENTIAL_API_DEFAULT_KEY}:
            for deduper in dedupers[SEQUENTIAL_API_DEFAULT_KEY]:
                dupe.apply(deduper)
        else:
            dupe.apply(dedupers)  # type: ignore[arg-type]

        df = dupe.canonicalize(
            columns,
            keep=keep,
            drop_duplicates=drop_duplicates,
            id=id,
        ).collect()  # type: ignore[assignment]

        if drop_canonical_id:
            return df.drop(columns=[CANONICAL_ID])
        return df
