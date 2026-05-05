from typing import Protocol

import pandas as pd

from liken.core.deduper import BaseDeduper
from liken.dedupers.cosine import cosine
from liken.dedupers.fuzzy import fuzzy
from liken.dedupers.jaccard import jaccard
from liken.dedupers.lsh import lsh
from liken.dedupers.tfidf import tfidf
from liken.liken import Dedupe
from liken.liken import dedupe
from liken.types import Columns
from liken.types import Keep


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
