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
Spark DataFrame, the wrapper will call the SparkDF class which will create
canonical IDs. However the output to this is RDDs which are then processed
by the executor into Spark Rows which are dispatched to worker nodes. Spark
Rows can be fully recovered to a Spark DataFrame using the same SparkDF class.

TODO:
    - CanonicalIdMixin should be defined first when inherited
    - A full interface can then be defined
"""

from __future__ import annotations

from typing import Self
from typing import final

import pandas as pd
import pyarrow as pa

from liken._constants import CANONICAL_ID
from liken._types import Keep
from liken.core.wrapper import DF
from liken.core.wrapper import CanonicalIdMixin


@final
class PandasDF(DF[pd.DataFrame], CanonicalIdMixin):
    """Pandas DataFrame wrapper"""

    def __init__(self, df: pd.DataFrame, id: str | None = None):
        self._df: pd.DataFrame = self._add_canonical_id(df, id)
        self._id = id

    # CANONICAL ID HELPERS:

    def _df_as_is(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _df_overwrite_id(self, df: pd.DataFrame, id: str) -> pd.DataFrame:
        return df.assign(**{CANONICAL_ID: df[id]})

    def _df_copy_id(self, df: pd.DataFrame, id: str) -> pd.DataFrame:
        return df.assign(**{CANONICAL_ID: df[id]})

    def _df_autoincrement_id(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(**{CANONICAL_ID: pd.RangeIndex(start=0, stop=len(df))})

    def _column_labels_list(self, df: pd.DataFrame) -> list[str]:
        return list(df.columns)

    # ARROW INTERFACES:

    def _get_col(self, column: str) -> pa.Array:
        return pa.array(self._df[column])

    def _get_cols(self, columns: tuple[str, ...]) -> pa.Table:
        return pa.Table.from_pandas(self._df[list(columns)])

    # WRAPPER METHODS:

    def put_col(self, column: str, array: list) -> Self:
        self._df = self._df.assign(**{column: array})
        return self

    def drop_col(self, column: str) -> Self:
        self._df = self._df.drop(columns=column)
        return self

    def drop_duplicates(self, keep: Keep) -> Self:
        self._df = self._df.drop_duplicates(keep=keep, subset=CANONICAL_ID)
        return self

    # SYNTHETIC RECORD:

    def synthesize_record(self) -> pd.DataFrame:
        def _first_non_null(series: pd.Series):
            non_null = series.dropna()
            return non_null.iloc[0] if not non_null.empty else pd.NA

        return self._df.groupby(CANONICAL_ID, as_index=False).agg(_first_non_null)
