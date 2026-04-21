from typing import Self

import pandas as pd

from liken._dedupers import fuzzy
from liken._dedupers import lsh
from liken._dedupers import tfidf
from liken._types import Columns
from liken._types import Keep
from liken.liken import Dedupe
from liken.liken import dedupe


# TODO: replace *args and **kwargs with actual params for better IDE autocompletion


def register_pd_affordances():

    @pd.api.extensions.register_dataframe_accessor("lk")
    class LikenAccessor:  # noqa
        def __init__(self, df: pd.DataFrame):
            self._deduper: Dedupe = dedupe(df)

        def fuzzy(self, *args, **kwargs) -> Self:
            self._deduper = self._deduper.apply(fuzzy(*args, **kwargs))
            return self

        def tfidf(self, *args, **kwargs) -> Self:
            self._deduper = self._deduper.apply(tfidf(*args, **kwargs))
            return self

        def lsh(self, *args, **kwargs) -> Self:
            self._deduper = self._deduper.apply(lsh(*args, **kwargs))
            return self

        def drop_duplicates(
            self,
            columns: Columns | None = None,
            *,
            keep: Keep = "first",
        ) -> pd.DataFrame:

            return self._deduper.drop_duplicates(
                columns=columns,
                keep=keep,
            )  # type: ignore
