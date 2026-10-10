"""Base DataFrame wrapper defining the uniform wrapper interface across backends."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Generic
from typing import Protocol
from typing import TypeVar

from liken.constants import CANONICAL_ID


if TYPE_CHECKING:
    import pyarrow as pa

    from liken.types import Columns


# TYPES


D = TypeVar("D")  # dataframe


# BASE


class DF(Generic[D]):
    """Base class defining a dataframe wrapper

    Defines inheritable methods as well as some of the interface.
    """

    def __init__(self, df: D):
        self._df: D = df

    def unwrap(self) -> D:
        return self._df

    def __getattr__(self, name: str) -> Any:
        """Delegation: use ._df without using property explicitly.

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

    def get_array(self, columns: Columns) -> pa.Array | pa.Table:
        """Generalise the getting of a df's column, or columns, to an array."""
        if isinstance(columns, str):
            return self._get_col(columns)
        return self._get_cols(columns)

    def get_canonical(self) -> pa.Array:
        """Convenience method"""
        return self.get_array(CANONICAL_ID)

    def synthesize_record(self) -> D:
        raise NotImplementedError


# CANONICAL ID


class AddsCanonical(Protocol[D]):
    """Mixin protocol"""

    def _df_as_is(self, df: D) -> D: ...
    def _df_overwrite_id(self, df: D, id: str) -> D: ...
    def _df_copy_id(self, df: D, id: str) -> D: ...
    def _df_autoincrement_id(self, df: D) -> D: ...
    def _column_labels_list(self, df: D) -> list[str]: ...


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
