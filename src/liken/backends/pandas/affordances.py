from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol
from typing import cast

import pandas as pd

from liken.dedupers.cosine import cosine
from liken.dedupers.fuzzy import fuzzy
from liken.dedupers.jaccard import jaccard
from liken.dedupers.lsh import lsh
from liken.dedupers.tfidf import tfidf
from liken.liken import Dedupe
from liken.liken import dedupe


if TYPE_CHECKING:
    from collections.abc import Callable

    from liken.core.deduper import BaseDeduper
    from liken.types import Columns
    from liken.types import Keep
    from liken.types import UserDataFrame


class DeduperProtocol(Protocol):
    @staticmethod
    def func(**kwargs: Any) -> BaseDeduper: ...


class DropMixin(DeduperProtocol):
    _df: pd.DataFrame

    def drop_duplicates(
        self,
        columns: Columns | None = None,
        *,
        keep: Keep = "first",
        **kwargs: Any,
    ) -> pd.DataFrame:

        deduper: Dedupe = dedupe(self._df)
        result: UserDataFrame = deduper.apply(
            self.func(**kwargs),
        ).drop_duplicates(
            columns=columns,
            keep=keep,
        )
        return cast("pd.DataFrame", result)


class Accessor(DropMixin):
    def __init__(self, df: pd.DataFrame):
        self._df: pd.DataFrame = df


def register_pd_affordances() -> None:

    def make_accessor(name: str, fn: Callable[..., BaseDeduper]) -> None:
        @pd.api.extensions.register_dataframe_accessor(name)
        class _Accessor(Accessor):
            @staticmethod
            def func(**kwargs: Any) -> BaseDeduper:
                return fn(**kwargs)

    make_accessor("fuzzy", fuzzy)
    make_accessor("tfidf", tfidf)
    make_accessor("lsh", lsh)
    make_accessor("cosine", cosine)
    make_accessor("jaccard", jaccard)
