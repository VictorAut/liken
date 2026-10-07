"""Ray DataFrame wrapper"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Self
from typing import final

import pandas as pd
import pyarrow as pa

from liken.constants import CANONICAL_ID
from liken.core.wrapper import DF
from liken.core.wrapper import CanonicalIdMixin


if TYPE_CHECKING:
    from ray.data import Dataset as RayDataset

    from liken.types import Keep


@final
class RayDF(DF["RayDataset"], CanonicalIdMixin):
    """Ray Dataset wrapper"""

    def __init__(self, df: RayDataset, id: str | None = None):
        import ray

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
        import ray

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
            for i, batch in enumerate(df.iter_batches(batch_format="pandas")):
                offset = offsets[i]
                batch = batch.reset_index(drop=True)
                batch[CANONICAL_ID] = range(offset, offset + len(batch))
                yield batch

        return ray.data.from_pandas_refs([ray.put(batch) for batch in _add_id_generator()])

    def _column_labels_list(self, df: RayDataset) -> list[str]:
        return df.columns()

    # ARROW INTERFACES:

    def _get_col(self, column: str) -> pa.Array:
        data = [row[column] for row in self._df.iter_rows()]
        return pa.array(data)

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

        return self._df.map_batches(fn, batch_format="pandas")
