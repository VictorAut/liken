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

from typing import TYPE_CHECKING
from typing import Self
from typing import final

import pandas as pd
import pyarrow as pa

from liken.constants import CANONICAL_ID
from liken.core.wrapper import DF
from liken.core.wrapper import CanonicalIdMixin


if TYPE_CHECKING:
    import dask.dataframe as dd

    from liken.types import Keep


@final
class DaskDF(DF["dd.DataFrame"], CanonicalIdMixin):
    """Dask DataFrame wrapper"""

    def __init__(self, df: dd.DataFrame, id: str | None = None, preserve_schema: bool = False):
        if preserve_schema:
            self._df = df
        else:
            self._df = self._add_canonical_id(df, id)
        self._id = id

    # CANONICAL ID HELPERS:

    def _df_as_is(self, df: dd.DataFrame) -> dd.DataFrame:
        return df

    def _df_overwrite_id(self, df: dd.DataFrame, id: str) -> dd.DataFrame:
        return df.assign(**{CANONICAL_ID: df[id]})

    def _df_copy_id(self, df: dd.DataFrame, id: str) -> dd.DataFrame:
        return df.assign(**{CANONICAL_ID: df[id]})

    def _df_autoincrement_id(self, df: dd.DataFrame) -> dd.DataFrame:

        lengths = df.map_partitions(len).compute()
        offsets = []
        total = 0
        for length in lengths:
            offsets.append(total)
            total += length

        def _assign(partition, partition_info=None):
            i = partition_info["number"]
            offset = offsets[i]
            partition = partition.reset_index(drop=True)
            partition[CANONICAL_ID] = range(offset, offset + len(partition))
            return partition

        meta = self._new_meta(df)

        return df.map_partitions(_assign, meta=meta)

    def _new_meta(self, df: dd.DataFrame, id: str | None = None) -> pd.DataFrame:
        meta = df._meta.copy()

        dtype = meta[id].dtype if id else "int64"

        meta[CANONICAL_ID] = pd.Series(dtype=dtype)

        return meta

    def _column_labels_list(self, df: dd.DataFrame) -> list[str]:
        return df.columns

    # ARROW INTERFACES:

    def _get_col(self, column: str) -> pa.Array:
        return pa.array(self._df[column].compute())

    def _get_cols(self, columns: tuple[str, ...]) -> pa.Table:
        return pa.Table.from_pandas(self._df[list(columns)].compute())

    # WRAPPER METHODS:

    def put_col(self, column: str, array: list) -> Self:
        self._df = self._df.assign(**{column: array})
        return self

    def drop_col(self, column: str) -> Self:
        self._df = self._df.drop(columns=column)
        return self

    def drop_duplicates(self, keep: Keep) -> Self:
        self._df = self._df.drop_duplicates(subset=CANONICAL_ID, keep=keep)
        return self

    # SYNTHETIC RECORD:

    def synthesize_record(self) -> dd.DataFrame:

        def _first_non_null(series):
            non_null = series.dropna()
            return non_null.iloc[0] if len(non_null) else None

        df = self._df.compute()
        return df.groupby(CANONICAL_ID).agg(_first_non_null).reset_index()