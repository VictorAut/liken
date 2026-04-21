from typing import Protocol

import pandas as pd

from liken._dedupers import BaseDeduper
from liken._dedupers import cosine
from liken._dedupers import fuzzy
from liken._dedupers import jaccard
from liken._dedupers import lsh
from liken._dedupers import tfidf
from liken._types import Columns
from liken._types import Keep
from liken.liken import Dedupe
from liken.liken import dedupe


class DeduperProtocol(Protocol):
    @staticmethod
    def func(**kwargs) -> BaseDeduper: ...


class DropMixin(DeduperProtocol):
    def drop_duplicates(
        self,
        columns: Columns | None = None,
        *,
        keep: Keep = "first",
        **kwargs,
    ) -> pd.DataFrame:

        self._deduper: Dedupe

        return self._deduper.apply(
            self.func(**kwargs),
        ).drop_duplicates(
            columns=columns,
            keep=keep,
        )


class Accessor(DropMixin):
    def __init__(self, df: pd.DataFrame):
        self._deduper: Dedupe = dedupe(df)


def register_pd_affordances():

    def make_accessor(name: str, fn) -> None:
        @pd.api.extensions.register_dataframe_accessor(name)
        class _Accessor(Accessor):  # noqa
            @staticmethod
            def func(**kwargs):
                return fn(**kwargs)

    make_accessor("fuzzy", fuzzy)
    make_accessor("tfidf", tfidf)
    make_accessor("lsh", lsh)
    make_accessor("cosine", cosine)
    make_accessor("jaccard", jaccard)
