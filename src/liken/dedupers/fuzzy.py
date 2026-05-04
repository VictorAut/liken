"""fuzzy deduper"""

from collections.abc import Iterator
from typing import Callable
from typing import ClassVar
from typing import Literal
from typing import final

import pyarrow as pa
from rapidfuzz import fuzz
from rapidfuzz import process

from liken.core.deduper import BaseDeduper
from liken.core.deduper import SingleColumnMixin
from liken.core.deduper import ThresholdDeduper
from liken.core.registries import dedupers_registry
from liken.types import SimilarPairIndices


@final
class Fuzzy(
    SingleColumnMixin,
    ThresholdDeduper,
):
    """
    Fuzzy string matching deduper
    """

    _NAME: ClassVar[str] = "fuzzy"

    _SCORERS: dict[str, Callable] = {
        "simple_ratio": fuzz.ratio,
        "partial_ratio": fuzz.partial_ratio,
        "token_sort_ratio": fuzz.token_sort_ratio,
        "token_set_ratio": fuzz.token_set_ratio,
        "weighted_ratio": fuzz.WRatio,
        "quick_ratio": fuzz.QRatio,
    }

    def __init__(
        self,
        threshold: float = 0.95,
        scorer: Literal[
            "simple_ratio",
            "partial_ratio",
            "token_sort_ratio",
            "token_set_ratio",
            "weighted_ratio",
            "quick_ratio",
        ] = "simple_ratio",
    ):
        super().__init__(
            threshold=threshold,
            scorer=scorer,
        )
        self._threshold = threshold
        self._scorer = scorer

    def get_scorer(self):
        return self._SCORERS.get(self._scorer, fuzz.ratio)

    def _gen_similarity_pairs(self, array: pa.Array) -> Iterator[SimilarPairIndices]:
        array: list = array.to_pylist()
        n = len(array)

        threshold = 100 * self._threshold

        scorer = self.get_scorer()

        for i, s1 in enumerate(array):
            if i + 1 >= n:
                break

            scores = process.cdist(
                [s1],
                array[i + 1 :],
                scorer=scorer,
            )[0]

            for offset, score in enumerate(scores):
                if score > threshold:
                    yield i, i + 1 + offset

    def __str__(self):
        return self.str_representation(self._NAME)


@dedupers_registry.register("fuzzy")
def fuzzy(
    threshold: float = 0.95,
    scorer: Literal[
        "simple_ratio",
        "partial_ratio",
        "token_sort_ratio",
        "token_set_ratio",
        "weighted_ratio",
        "quick_ratio",
    ] = "simple_ratio",
) -> BaseDeduper:
    """Near string deduplication.

    Usage is on single columns of a dataframe.

    Args:
        threshold: The minimum threshold at which similarity between two pairs
            of values will be considered valid for deduplication.
        scorer: The fuzzy scorer. Defaults to "simple ratio". Options are
            "simple_ratio", "partial_ratio", "token_sort_ratio",
            "token_set_ratio", "weighted_ratio", "quick_ratio".

    Returns:
        Instance of `BaseDeduper`.

    Example:
        Applied to a single column:

            import liken as lk

            df = (
                lk.dedupe(df)
                .apply({"address": fuzzy(threshold=0.8)})
                .drop_duplicates(keep="last")
            )

        E.g.

            >>> df # Before
            +------+-----------+----------------------+
            | id   |  address  |         email        |
            +------+-----------+----------------------+
            |  1   |  london   |  fizzpop@gmail.com   |
            |  2   |   null    |  foobar@gmail.com    |
            |  3   |  london   |  foobar@gmail.co.uk  |
            +------+-----------+----------------------+

            >>> df # After
            +------+-----------+----------------------+
            | id   |  address  |         email        |
            +------+-----------+----------------------+
            |  1   |  london   |  fizzpop@gmail.com   |
            |  3   |  london   |  foobar@gmail.co.uk  |
            +------+-----------+----------------------+
    """
    return Fuzzy(threshold=threshold, scorer=scorer)
