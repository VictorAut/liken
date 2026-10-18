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

    from liken.backends.ray.wrapper import RayDF
    from liken.collections.base import Pipeline
    from liken.collections.dict import DeduplicationDict
    from liken.types import Columns
    from liken.types import Keep


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
        source_id: str | None = None,
        canonical_id: str = CANONICAL_ID,
    ) -> RayDF:
        """Maps dataframe partitions to be processed by Ray enforced to use
        pandas for each partition.
        """

        from liken.backends.ray.wrapper import RayDF
        from liken.liken import Dedupe

        def _process_batch(batch: pd.DataFrame) -> pd.DataFrame:
            from liken.collections.base import SEQUENTIAL_API_DEFAULT_KEY
            from liken.collections.dict import DeduplicationDict as DictImpl

            dupe = Dedupe(batch)  # type: ignore[return-value]

            # Internal handoff: the sequential API stores dedupers under the
            # reserved `_default_` key, which a public dict apply must not use.
            # Re-apply them individually to keep the sequential semantics.
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
            return dupe.collect()  # type: ignore[return-value]

        # IMPORTANT: "pandas" batch
        # canonical_id is passed to keep the schema-only re-wrap from adding a
        # second canonical column under the default label.
        df = RayDF(df._df.map_batches(_process_batch, batch_format="pandas"), canonical_id=canonical_id)  # type: ignore[arg-type]

        if drop_canonical_id:
            return df.drop_col(canonical_id)
        return df
