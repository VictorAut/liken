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

from typing import Any
from typing import Generic
from typing import Protocol
from typing import TypeVar

import pyarrow as pa
from pyarrow.compute import coalesce

from liken._constants import CANONICAL_ID
from liken._constants import NA_PLACEHOLDER
from liken._types import Columns


# TYPES


D = TypeVar("D")  # dataframe


# BASE


class DF(Generic[D]):
    """Base class defining a dataframe wrapper

    Defines inheritable methods as well as some of the interface

    TODO:
        - define a protocol interface
        - tighten generics
    """

    def __init__(self, df: D):
        self._df: D = df

    def unwrap(self) -> D:
        return self._df

    def __getattr__(self, name: str) -> Any:
        """Delegation: use ._df without using property explicitely.

        So, the use of Self even with no attribute returns ._df attribute.
        Therefore calling Self == call Self._df. This is useful as it makes the
        API more concise in other modules.

        For example, as the Dedupe class attribute ._df is an instance of this
        class, it avoids having to do Dedupe()._df._df to access the actual
        dataframe.
        """
        return getattr(self._df, name)

    def _get_col(self, column: str) -> pa.Array:
        del column
        raise NotImplementedError

    def _get_cols(self, columns: tuple[str, ...]) -> pa.Table:
        del columns
        raise NotImplementedError

    def get_array(self, columns: Columns, with_na: bool = False) -> pa.Array | pa.Table:
        """Generalise the getting of a column, or columns of a df to an array.

        Handles single column and multicolumn. For instances of single column
        the initial column can initially be filled null placeholders, to allow
        for use by dedupers. This is optional so that specific dedupers
        that do care about nulls are not affected (e.g. IsNA).
        """
        if isinstance(columns, str):
            col: pa.Array = self._get_col(columns)
            if with_na:
                return coalesce(col, NA_PLACEHOLDER)
            return col
        return self._get_cols(columns)

    def get_canonical(self) -> pa.Array:
        """Convenience method"""
        return self.get_array(CANONICAL_ID)

    def synthesize_record(self) -> D:
        raise NotImplementedError


# CANONICAL ID


class AddsCanonical(Protocol):
    """Mixin protocol"""

    def _df_as_is(self, df): ...
    def _df_overwrite_id(self, df, id: str): ...
    def _df_copy_id(self, df, id: str): ...
    def _df_autoincrement_id(self, df): ...
    def _column_labels_list(self, df): ...


class CanonicalIdMixin(AddsCanonical):
    """Defines creation of canonical id upon wrapping a dataframe

    By default a canonical ID is an auto-incrementing numeric field, starting
    from zero.

    However, the canonical ID field can also be:
        - already present in the dataframe as "canonical_id"
        - copied from another "id" field

    In those other instances the resultant canonical id field can therefore
    also be a string field.
    """

    def _add_canonical_id(self, df, id: str | None):

        has_canonical: bool = CANONICAL_ID in self._column_labels_list(df)
        id_is_canonical: bool = id == CANONICAL_ID

        if has_canonical:
            if id:
                if id_is_canonical:
                    return self._df_as_is(df)
                # overwrite with id
                return self._df_overwrite_id(df, id)
            return self._df_as_is(df)
        if id:
            # write new with id
            return self._df_copy_id(df, id)
        # write new auto-incrementing
        return self._df_autoincrement_id(df)
