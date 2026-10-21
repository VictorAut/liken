"""sorted neighbourhood deduper"""

from collections.abc import Callable
from collections.abc import Iterator
from typing import Any
from typing import ClassVar
from typing import final

import pyarrow as pa
from rapidfuzz import process

from liken.core.deduper import BaseDeduper
from liken.core.deduper import SingleColumnMixin
from liken.core.deduper import ThresholdDeduper
from liken.core.missing import partition_missing
from liken.core.registries import dedupers_registry
from liken.dedupers.fuzzy import _BOUNDARY_TOLERANCE
from liken.dedupers.fuzzy import _SCORER_NAMES
from liken.dedupers.fuzzy import _SCORERS
from liken.dedupers.fuzzy import ScorerName
from liken.types import SimilarPairIndices
from liken.validations import require_choice
from liken.validations import require_int
from liken.validations import require_type


@final
class SortedNeighbourhood(
    SingleColumnMixin,
    ThresholdDeduper,
):
    """
    Sorted-neighbourhood-method deduper

    An unknown scorer name is rejected at construction.
    """

    _NAME: ClassVar[str] = "sorted_neighbourhood"

    _SCORERS: ClassVar[dict[str, Callable]] = _SCORERS

    def __init__(
        self,
        threshold: float = 0.95,
        scorer: ScorerName = "simple_ratio",
        *,
        window: int,
    ):
        require_type("scorer", scorer, str)
        require_choice("scorer", scorer, _SCORER_NAMES)
        window = require_int("window", window, low=2)
        super().__init__(threshold=threshold, scorer=scorer, window=window)
        self._threshold = threshold
        self._scorer = scorer
        self._window = window

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

        # stable sort of the present positions on the raw value: code-point
        # order, deterministic across backends; ties keep input order
        ordered: list[int] = sorted(present, key=values.__getitem__)

        n = len(ordered)

        for pos in range(n):
            window = ordered[pos + 1 : pos + self._window]

            if not window:
                break

            # the row's value against its forward window, grouped by the
            # pair's original-index direction: fuzzy scores each pair
            # (earlier original index -> later original index), and the
            # subset-of-fuzzy claim must hold for the asymmetric scorers
            # too (partial_ratio).
            s1 = values[ordered[pos]]
            later = [i for i in window if i > ordered[pos]]
            earlier = [i for i in window if i < ordered[pos]]

            if later:
                scores: Any = process.cdist([s1], [values[i] for i in later], scorer=scorer)[0]
                for offset, score in enumerate(scores):
                    if score >= threshold - _BOUNDARY_TOLERANCE:
                        a, b = ordered[pos], later[offset]
                        yield (a, b) if a < b else (b, a)

            if earlier:
                # the window partner's value is the query: fuzzy's
                # direction is (earlier original index -> later), so the
                # partner is the query and this row the choice
                scores = process.cdist([values[i] for i in earlier], [s1], scorer=scorer)[:, 0]
                for offset, score in enumerate(scores):
                    if score >= threshold - _BOUNDARY_TOLERANCE:
                        a, b = earlier[offset], ordered[pos]
                        yield (a, b) if a < b else (b, a)

    def __str__(self):
        return self.str_representation(self._NAME)


@dedupers_registry.register("sorted_neighbourhood")
def sorted_neighbourhood(
    threshold: float = 0.95,
    scorer: ScorerName = "simple_ratio",
    *,
    window: int,
) -> BaseDeduper:
    """Sorted-neighbourhood-method deduplication.

    Usage is on single columns of a dataframe. Appropriate for string
    data sorted close to its duplicate axis.

    The column is sorted by value and each row is scored only against
    the following `window - 1` rows in sorted order; a pair matches
    when its score reaches the threshold. The scored pair set is a
    strict subset of all-pairs `fuzzy` at the same threshold and
    scorer — same acceptance semantics, restricted pair generation.

    Sorting is on the raw values (code-point order, no locale). Two
    values similar in the middle but different at the start, such as
    postcodes "OL5 9PL" and "0L5 9PL", can fall outside each other's
    windows. Word-order variants ("John Smith" vs "Smith John") land
    far apart in sort order and are not this deduper's job — that is
    `sorted_tokens`. What the window catches is typo-drift between
    values that are close in sorted order.

    Each row keeps at most `window - 1` candidate scores instead of a
    score against every other value: scorer work falls from O(n^2) to
    O(n * window). Values whose sort keys are equal (identical
    strings) sit in input order within the sorted run, and a tie group
    larger than the window is cut at window edges: some identical
    pairs are never scored, and which pairs those are depends on input
    order. The same input order gives the same output; reordering the
    column can change the result.

    `sorted_neighbourhood(threshold, scorer, window=w)` differs from
    `fuzzy(threshold, scorer, topn=k)`: the neighbourhood is a bounded
    slice of the sorted order around each value, while `topn` keeps
    each value's k best partners anywhere in the column. Neither
    subsumes the other. Missing values are unaffected by the window:
    they pair with each other star-shaped, never with a value, exactly
    as in `fuzzy`.

    Args:
        threshold: The minimum threshold at which similarity between two pairs
            of values will be considered valid for deduplication.
        scorer: The fuzzy scorer, shared with `fuzzy`. Defaults to
            "simple ratio". Options are
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
        window: The number of neighbouring rows, in sorted order, each
            value is scored against. Required: there is no principled
            default width. The comparison is inclusive: a pair scoring
            exactly at the threshold matches.

    Returns:
        Instance of `BaseDeduper`.

    Example:
        Applied to a single column:

            import liken as lk

            df = (
                lk.dedupe(df)
                .apply({"address": sorted_neighbourhood(threshold=0.8, window=5)})
                .drop_duplicates(keep="last")
            )
    """
    return SortedNeighbourhood(threshold=threshold, scorer=scorer, window=window)
