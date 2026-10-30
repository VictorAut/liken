"""fuzzy deduper"""

from collections.abc import Callable
from collections.abc import Iterator
from typing import Any
from typing import ClassVar
from typing import Final
from typing import Literal
from typing import TypeAlias
from typing import final

import numpy as np
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


def _select_top(
    matrix: np.ndarray,
    base: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row top-`width` columns by (score desc, index asc).

    Selection over one strip matrix: the running top-k state's tie rule
    must match `process.extract`'s documented "ties sort by choice
    index". A full argsort per strip is O(rows x cols x log cols) and
    dominates wall-clock; argpartition is O(cols) but picks arbitrary
    indices among ties at the cut. The shape here: argpartition to
    `width` candidates, then repair ties at the boundary — candidates
    strictly above the partition's minimum score sort by (score desc,
    index asc) within the small selected slice, and columns tied at the
    boundary fill the remaining slots in ascending index order.

    Args:
        matrix: Score matrix `(rows, cols)`; float64.
        base: Global index of column 0 (strip column offsets differ
            per direction).
        width: Slots to return per row, at most `cols`.

    Returns:
        `(indices, scores)` of shape `(rows, width)`: global partner
        indices and their scores, `-1` / `-inf` padded.
    """
    rows, cols = matrix.shape
    w = min(width, cols)

    if cols > w:
        part = np.argpartition(-matrix, w - 1, axis=1)[:, :w]
        part_score = np.take_along_axis(matrix, part, axis=1)
        boundary = part_score.min(axis=1, keepdims=True)
        # the cut ties: every column whose score equals the boundary
        # competes with the partition's tail by index
        tied = matrix == boundary
        part_order = np.lexsort((part, -part_score), axis=1)
        part_sorted_idx = np.take_along_axis(part, part_order, axis=1) + base
        part_sorted_sco = np.take_along_axis(part_score, part_order, axis=1)
        # strictly-above candidates are all within the partition; the
        # boundary is its minimum, so at most w - 1 of them
        n_above = (part_score > boundary).sum(axis=1)
        out_idx = np.full((rows, width), -1, dtype=np.int64)
        out_sco = np.full((rows, width), -np.inf, dtype=np.float64)
        for r in range(rows):
            na = int(n_above[r])
            out_idx[r, :na] = part_sorted_idx[r, :na]
            out_sco[r, :na] = part_sorted_sco[r, :na]
            need = width - na
            if need > 0:
                tied_cols = np.flatnonzero(tied[r])[:need]
                n_t = len(tied_cols)
                out_idx[r, na : na + n_t] = tied_cols + base
                out_sco[r, na : na + n_t] = boundary[r, 0]
        return out_idx, out_sco

    order = np.argsort(-matrix, axis=1, kind="stable")
    idx = np.take_along_axis(np.broadcast_to(np.arange(cols), (rows, cols)), order, axis=1) + base
    sco = np.take_along_axis(matrix, order, axis=1)
    if w < width:
        idx = np.pad(idx, ((0, 0), (0, width - w)), constant_values=-1)
        sco = np.pad(sco, ((0, 0), (0, width - w)), constant_values=-np.inf)
    return idx, sco


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
        and the score reaches the threshold.

        Pairs come from the block machinery: `process.cdist` strips of
        `block_rows` queries against the remaining tail, sized by
        `block_rows_for` under the `max_block_memory` cap and scored with
        `resolve_workers`' worker count — the same shape as the all-pairs
        path. The strip's `score_cutoff` zeroes below-cutoff cells before
        materialization, so candidates are the above-cutoff cells only;
        both strip directions append their candidates as (owner, index,
        score) triples and one global sort by (owner, score desc, index
        asc) reduces them — no per-strip state merge. A strip too dense
        to buffer sparsely contributes bounded per-row top-`width`
        windows instead (`_select_top`), which compose exactly: the
        top-k of a union is the k best of the union of the parts' top-ks.

        Ranking is score descending then index ascending — the tie rule
        `process.extract` documents. Within an owner's sorted slice,
        duplicates (one pair can arrive from both directions with the
        identical score) and self drop by index, the first `topn`
        partners remain, and the Python-side cutoff re-check with the
        boundary tolerance stays authoritative regardless of rapidfuzz
        version behaviour.
        """
        n = len(scored)
        if n < 2:
            return

        topn = self._topn
        cutoff: float = threshold - _BOUNDARY_TOLERANCE
        width = topn + 1  # window width: k partners + self transiently

        max_block_memory = DEFAULT_BLOCK_MEMORY if self._max_block_memory is None else int(self._max_block_memory)
        block_rows = block_rows_for(n, probe_itemsize(scorer, dtype="float64"), max_block_memory)
        workers = resolve_workers(self._workers, distributed=self.distributed)

        # a sparse strip's triples cost 24 bytes each; the budget bounds
        # them by the same cap as the matrix, with the window path as the
        # floor for degenerate caps
        dense_budget = max(max_block_memory // 24, block_rows * width)

        parts_owner: list[np.ndarray] = []
        parts_index: list[np.ndarray] = []
        parts_score: list[np.ndarray] = []

        start = 0
        while start < n - 1:
            end = min(start + block_rows, n)
            strip = np.asarray(
                process.cdist(
                    scored[start:end],
                    scored[start + 1 :],
                    scorer=scorer,
                    dtype="float64",
                    score_cutoff=cutoff,
                    workers=workers,
                )
            )
            _, c = strip.shape

            if np.count_nonzero(strip) > dense_budget:
                q_idx, q_sco = _select_top(strip, start + 1, width)
                parts_owner.append(np.repeat(np.arange(start, end), width))
                parts_index.append(q_idx.ravel())
                parts_score.append(q_sco.ravel())
                c_idx, c_sco = _select_top(strip.T, start, width)
                parts_owner.append(np.repeat(np.arange(start + 1, start + 1 + c), width))
                parts_index.append(c_idx.ravel())
                parts_score.append(c_sco.ravel())
            else:
                rows_idx, cols_idx = np.nonzero(strip)
                scores = strip[rows_idx, cols_idx]
                # query side: rows start..end-1 gain partners start+1..n-1
                parts_owner.append(rows_idx + start)
                parts_index.append(cols_idx + start + 1)
                parts_score.append(scores)
                # column side: rows start+1..n-1 gain partners start..end-1
                parts_owner.append(cols_idx + start + 1)
                parts_index.append(rows_idx + start)
                parts_score.append(scores)

            start = end

        if not parts_owner:
            return

        owner = np.concatenate(parts_owner)
        index = np.concatenate(parts_index)
        score = np.concatenate(parts_score)

        # one global reduction: owner ascending (primary), score
        # descending, index ascending within ties
        order = np.lexsort((index, -score, owner))
        owner = owner[order]
        index = index[order]
        score = score[order]

        seen: set[tuple[int, int]] = set()
        bounds = np.searchsorted(owner, np.arange(n), side="right")
        lo = 0
        for pos in range(n):
            hi = bounds[pos]
            if hi == lo:
                continue
            partners = 0
            prev_index = -1
            for slot in range(lo, hi):
                if partners == topn:
                    break
                idx = int(index[slot])
                if idx == prev_index:  # both directions carry the pair
                    continue
                prev_index = idx
                if idx == pos:
                    continue
                partners += 1
                if score[slot] < cutoff:
                    continue
                pair = (min(present[pos], present[idx]), max(present[pos], present[idx]))
                if pair not in seen:
                    seen.add(pair)
                    yield pair
            lo = hi

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
        topn: The number of best matches to consider per value. Each value
            holds at most `k` candidate scores: a pair matches when either
            side holds the other in its k best and the score reaches the
            threshold. The default `None` compares every value against
            every other and keeps every matching pair. Scorer work is
            unchanged in either mode; only the matches kept are limited,
            so `topn` bounds the pair output and downstream work, not the
            scoring cost. Matching is approximate: in a cluster of more
            than k near-identical records some pairs are dropped, so the
            cluster may fail to link transitively even though union-find
            merges whatever pairs survive. Ties at rank k are broken by
            row order: the same input order gives the same output, and
            reordering the column can change the result.
        workers: The number of worker threads used for scoring. This
            parameter has multiple effective defaults: `None` chooses
            automatically, using all CPU cores locally or for single node
            backends, but a single worker under a distributed backend.`-1`
            overrides the default and uses all cores, else choose an integer
            number for the desired number of cores.
        max_block_memory: The approximate memory budget, in bytes, of one
            scored block. Values are scored in
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
