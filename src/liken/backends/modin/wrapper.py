"""Modin DataFrame wrapper"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Self
from typing import final

import pyarrow as pa

from liken.constants import CANONICAL_ID
from liken.core.wrapper import DF
from liken.core.wrapper import CanonicalIdMixin


if TYPE_CHECKING:
    import modin.pandas as mpd

    from liken.types import Keep


@final
class ModinDF(DF["mpd.DataFrame"], CanonicalIdMixin):
    """Modin DataFrame wrapper"""

    def __init__(self, df: mpd.DataFrame, id: str | None = None):
        self._df: mpd.DataFrame = self._add_canonical_id(df, id)
        self._id = id

    # CANONICAL ID HELPERS:

    def _df_as_is(self, df: mpd.DataFrame) -> mpd.DataFrame:
        return df

    def _df_overwrite_id(self, df: mpd.DataFrame, id: str) -> mpd.DataFrame:
        return df.assign(**{CANONICAL_ID: df[id]})

    def _df_copy_id(self, df: mpd.DataFrame, id: str) -> mpd.DataFrame:
        return df.assign(**{CANONICAL_ID: df[id]})

    def _df_autoincrement_id(self, df: mpd.DataFrame) -> mpd.DataFrame:
        import modin.pandas as mpd

        return df.assign(**{CANONICAL_ID: mpd.RangeIndex(start=0, stop=len(df))})

    def _column_labels_list(self, df: mpd.DataFrame) -> list[str]:
        return df.columns

    # ARROW INTERFACES:

    def _get_col(self, column: str) -> pa.Array:
        return pa.array(self._df[column]._to_pandas())

    def _get_cols(self, columns: tuple[str, ...]) -> pa.Table:
        return pa.Table.from_pandas(
            self._df[list(columns)]._to_pandas()
        )  # TODO: there's a to_pandas() public function?

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

    def synthesize_record(self) -> mpd.DataFrame:
        def _first_non_null(series):
            non_null = series.dropna()
            return non_null.iloc[0] if not non_null.empty else None

        return self._df.groupby(CANONICAL_ID, as_index=False).agg(_first_non_null)
