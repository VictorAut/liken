"""fuzzy deduper"""

from collections.abc import Callable
from collections.abc import Iterator
from typing import ClassVar
from typing import Final
from typing import Literal
from typing import TypeAlias
from typing import final

import pyarrow as pa
from rapidfuzz import fuzz
from rapidfuzz import process
from rapidfuzz.distance import JaroWinkler

from liken.core.deduper import BaseDeduper
from liken.core.deduper import SingleColumnMixin
from liken.core.deduper import ThresholdDeduper
from liken.core.missing import partition_missing
from liken.core.registries import dedupers_registry
from liken.types import SimilarPairIndices
from liken.validations import require_choice
from liken.validations import require_type


ScorerName: TypeAlias = Literal[
    "simple_ratio",
    "partial_ratio",
    "token_sort_ratio",
    "token_set_ratio",
    "weighted_ratio",
    "quick_ratio",
    "jaro_winkler",
]


# `100 * threshold` drifts by a few double-precision ULPs for some
# two-decimal thresholds, which would make a pair scoring exactly the
# threshold match or not depending on the rounding direction. The
# comparison tolerates scores within this guard below the boundary. The
# drift it repairs is ~1e-14 and real distinct similarity scores differ by
# far more, so the guard only admits scores equal to the boundary at any
# meaningful precision.
_BOUNDARY_TOLERANCE: Final[float] = 1e-9


def _jaro_winkler(s1: str, s2: str, score_cutoff: float | None = None) -> float:
    """JaroWinkler normalized similarity on the 0-100 scale.

    `JaroWinkler.normalized_similarity` returns 0-1; the other scorers
    and the threshold comparison use 0-100, so it is scaled here at the
    registry boundary.

    `score_cutoff` is accepted and ignored: `process.cdist` passes it as
    `None` and a two-argument callable would raise `TypeError`.
    """
    return 100 * JaroWinkler.normalized_similarity(s1, s2)


_SCORERS: Final[dict[str, Callable]] = {
    "simple_ratio": fuzz.ratio,
    "partial_ratio": fuzz.partial_ratio,
    "token_sort_ratio": fuzz.token_sort_ratio,
    "token_set_ratio": fuzz.token_set_ratio,
    "weighted_ratio": fuzz.WRatio,
    "quick_ratio": fuzz.QRatio,
    "jaro_winkler": _jaro_winkler,
}

_SCORER_NAMES: Final[tuple[str, ...]] = tuple(_SCORERS)


@final
class Fuzzy(
    SingleColumnMixin,
    ThresholdDeduper,
):
    """
    Fuzzy string matching deduper

    An unknown scorer name is rejected at construction.
    """

    _NAME: ClassVar[str] = "fuzzy"

    _SCORERS: ClassVar[dict[str, Callable]] = _SCORERS

    def __init__(
        self,
        threshold: float = 0.95,
        scorer: ScorerName = "simple_ratio",
    ):
        require_type("scorer", scorer, str)
        require_choice("scorer", scorer, _SCORER_NAMES)
        super().__init__(
            threshold=threshold,
            scorer=scorer,
        )
        self._threshold = threshold
        self._scorer = scorer

    def get_scorer(self):
        return self._SCORERS[self._scorer]

    def _gen_similarity_pairs(self, array: pa.Array) -> Iterator[SimilarPairIndices]:
        values: list = array.to_pylist()

        missing, present = partition_missing(values)

        # star-shaped missing pairs: k missing values cost k-1 pairs
        for i in missing[1:]:
            yield missing[0], i

        if not present:
            return

        threshold = 100 * self._threshold

        scorer = self.get_scorer()

        # the scorer receives non-missing values only
        scored = [values[i] for i in present]

        for pos, s1 in enumerate(scored):
            if pos + 1 >= len(scored):
                break

            scores = process.cdist(
                [s1],
                scored[pos + 1 :],
                scorer=scorer,
            )[0]

            for offset, score in enumerate(scores):
                if score >= threshold - _BOUNDARY_TOLERANCE:
                    yield present[pos], present[pos + 1 + offset]

    def __str__(self):
        return self.str_representation(self._NAME)


@dedupers_registry.register("fuzzy")
def fuzzy(
    threshold: float = 0.95,
    scorer: ScorerName = "simple_ratio",
) -> BaseDeduper:
    """Near string deduplication.

    Usage is on single columns of a dataframe.

    Args:
        threshold: The minimum threshold at which similarity between two pairs
            of values will be considered valid for deduplication.
        scorer: The fuzzy scorer. Defaults to "simple ratio". Options are
            - "simple_ratio"
            - "partial_ratio"
            - "token_sort_ratio"
            - "token_set_ratio"
            - "weighted_ratio"
            - "quick_ratio"
            - "jaro_winkler": the Jaro-Winkler similarity, weighted
              towards shared prefixes. Scores on the same 0-100 scale as
              the other options. Suited to short strings such as names,
              postcodes and product codes.

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
