"""fuzzy deduper"""

from collections.abc import Callable
from collections.abc import Iterator
from typing import Any
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
from liken.dedupers._blockwise import DEFAULT_BLOCK_MEMORY
from liken.dedupers._blockwise import block_rows_for
from liken.dedupers._blockwise import probe_itemsize
from liken.dedupers._blockwise import resolve_workers
from liken.dedupers._blockwise import similar_pairs_blockwise
from liken.exceptions import InvalidArgumentValueError
from liken.types import SimilarPairIndices
from liken.validations import require_choice
from liken.validations import require_int
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
    `None` and a two-argument callable would raise `TypeError`. Under
    `topn`, `process.extract` passes a real cutoff and compares the score
    this wrapper returns against it; ignoring the cutoff inside the
    wrapper stays correct, because the tolerant filter in `_topk_pairs`
    is authoritative.
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
        topn: int | None = None,
        workers: int | None = None,
        max_block_memory: int | None = None,
    ):
        require_type("scorer", scorer, str)
        require_choice("scorer", scorer, _SCORER_NAMES)
        topn = require_int("topn", topn, low=1, allow_none=True)
        workers = require_int("workers", workers, low=-1, allow_none=True)
        if workers == 0:
            raise InvalidArgumentValueError("`workers` must be a nonzero integer or -1, got 0")
        max_block_memory = require_int("max_block_memory", max_block_memory, low=0, allow_none=True)
        init_kwargs: dict[str, Any] = {"threshold": threshold, "scorer": scorer}
        if topn is not None:
            init_kwargs["topn"] = topn
        if workers is not None:
            init_kwargs["workers"] = workers
        if max_block_memory is not None:
            init_kwargs["max_block_memory"] = max_block_memory
        super().__init__(**init_kwargs)
        self._threshold = threshold
        self._scorer = scorer
        self._topn = topn
        self._workers = workers
        self._max_block_memory = max_block_memory

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

        if self._topn is not None:
            yield from self._topk_pairs(present, scored, threshold, scorer)
            return

        max_block_memory = DEFAULT_BLOCK_MEMORY if self._max_block_memory is None else int(self._max_block_memory)
        block_rows = block_rows_for(len(scored), probe_itemsize(scorer, dtype=None), max_block_memory)
        workers = resolve_workers(self._workers, distributed=self.distributed)

        for r, c in similar_pairs_blockwise(
            scored,
            compare=scorer,
            low=False,
            bound=threshold - _BOUNDARY_TOLERANCE,
            block_rows=block_rows,
            workers=workers,
        ):
            yield present[r], present[c]

    def _topk_pairs(
        self,
        present: list[int],
        scored: list,
        threshold: float,
        scorer: Callable,
    ) -> Iterator[SimilarPairIndices]:
        """Candidate-limited pairs: each row keeps its k best partners.

        A pair is emitted when either side holds the other in its k best
        and the score reaches the threshold. Per-row `process.extract`
        over the full present-value list computes each row's directed
        top-k, so both directions are covered without a second pass.

        The cutoff passed to `process.extract` is inclusive and already
        applies the boundary tolerance, mirroring the all-pairs
        comparison; the Python-side re-check keeps that contract
        authoritative regardless of rapidfuzz version behaviour.
        """
        topn = self._topn
        cutoff: float = threshold - _BOUNDARY_TOLERANCE

        seen: set[tuple[int, int]] = set()

        for pos, s1 in enumerate(scored):
            results = process.extract(
                s1,
                scored,
                scorer=scorer,
                limit=topn + 1,
                score_cutoff=cutoff,
            )

            # drop self by index, keep at most topn partners; ties sort by
            # choice index, so self may sit beyond rank topn + 1 and the
            # truncation then takes the first topn non-self entries
            partners = 0
            for _choice, score, index in results:
                if partners == topn:
                    break
                if index == pos:
                    continue
                partners += 1
                if score < cutoff:
                    continue
                pair = (min(present[pos], present[index]), max(present[pos], present[index]))
                if pair not in seen:
                    seen.add(pair)
                    yield pair

    def __str__(self):
        return self.str_representation(self._NAME)


@dedupers_registry.register("fuzzy")
def fuzzy(
    threshold: float = 0.95,
    scorer: ScorerName = "simple_ratio",
    topn: int | None = None,
    workers: int | None = None,
    max_block_memory: int | None = None,
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
        topn: The number of best matches to consider per value. The
            headline value of `topn` is memory: each value holds at most
            `k` candidate scores instead of a score against every other
            value. The default `None` compares every value against every
            other. An integer `k` bounds the matches held per value to
            `k`: a pair matches when either side holds the other in its
            k best and the score reaches the threshold. Scorer work per
            value is unchanged; only the matches kept are limited.
            Matching is approximate: in a cluster of more than k near-identical
            records some pairs are dropped, so the cluster may fail to
            link transitively even though union-find merges whatever
            pairs survive. Ties at rank k are broken by row order: the
            same input order gives the same output, and reordering the
            column can change the result.
        workers: The number of worker threads used for scoring, in the
            all-pairs mode. This parameter has multiple effective
            defaults: `None` chooses automatically, using all CPU cores
            locally and a single worker under a distributed backend,
            where the executor owns parallelism and an all-cores request
            would oversubscribe the cores. An explicit integer always
            wins, `-1` meaning all cores. Under a distributed backend an
            explicit all-cores value risks executor contention.
        max_block_memory: The approximate memory budget, in bytes, of one
            scored block in the all-pairs mode. Values are scored in
            blocks bounded by this cap; smaller blocks lower peak memory
            at some cost in scoring speed. Defaults to a library constant
            of 64 MiB. A cap too small for even one row's score band
            clamps to a one-row block, so peak memory may exceed the cap
            by that band.

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
    return Fuzzy(threshold=threshold, scorer=scorer, topn=topn, workers=workers, max_block_memory=max_block_memory)
