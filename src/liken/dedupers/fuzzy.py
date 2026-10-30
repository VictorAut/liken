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


def _merge_topk(
    state_score: np.ndarray,
    state_index: np.ndarray,
    owners: np.ndarray,
    indices: np.ndarray,
    scores: np.ndarray,
    width: int,
) -> None:
    """Merge candidate `(owner, index, score)` triples into the state.

    Candidates scatter into per-owner buckets of `2 * width` slots (each
    owner receives at most `width` from either direction per strip),
    merged with the owner's existing state and re-selected to the top
    `width` by (score desc, index asc). A partner can arrive from both
    directions with the identical score, so duplicates deduplicate by
    index; re-selection is unconditional because the dedupe pass leaves
    candidates index-ordered while emission depends on score order.
    """
    order = np.argsort(owners, kind="stable")
    owners_sorted = owners[order]
    idx_sorted = indices[order]
    sco_sorted = scores[order]

    uniq, first_pos = np.unique(owners_sorted, return_index=True)
    group_sizes = np.diff(np.append(first_pos, len(owners_sorted)))
    pos_in_group = np.arange(len(owners_sorted)) - np.repeat(first_pos, group_sizes)
    flat_ok = pos_in_group < 2 * width
    slots = owners_sorted * 2 * width + pos_in_group

    bucket = np.full((len(state_score), 2 * width), -np.inf, dtype=np.float64)
    bucket_idx = np.full((len(state_score), 2 * width), -1, dtype=np.int64)
    bucket_flat = bucket.ravel()
    bucket_idx_flat = bucket_idx.ravel()
    bucket_flat[slots[flat_ok]] = sco_sorted[flat_ok]
    bucket_idx_flat[slots[flat_ok]] = idx_sorted[flat_ok]

    touched = uniq
    all_s = np.concatenate([state_score[touched], bucket[touched]], axis=1)
    all_i = np.concatenate([state_index[touched], bucket_idx[touched]], axis=1)

    keep = all_i >= 0
    all_s = np.where(keep, all_s, -np.inf)
    all_i = np.where(keep, all_i, -1)

    # dedupe by index: sort index-ascending (score descending within an
    # index — identical by symmetry, so the first occurrence is kept)
    dedupe_order = np.lexsort((-all_s, all_i), axis=1)
    all_s = np.take_along_axis(all_s, dedupe_order, axis=1)
    all_i = np.take_along_axis(all_i, dedupe_order, axis=1)
    dup = np.zeros_like(all_i, dtype=bool)
    dup[:, 1:] = all_i[:, 1:] == all_i[:, :-1]
    valid = (all_i >= 0) & ~dup
    all_s = np.where(valid, all_s, -np.inf)
    all_i = np.where(valid, all_i, -1)

    # re-select by (score desc, index asc), always
    final_order = np.lexsort((all_i, -all_s), axis=1)
    all_s = np.take_along_axis(all_s, final_order, axis=1)
    all_i = np.take_along_axis(all_i, final_order, axis=1)
    state_score[touched, :] = all_s[:, :width]
    state_index[touched, :] = all_i[:, :width]


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
        path. Each strip merges into a per-row running top-k state of
        `k + 1` slots, from both directions: the query rows gain the
        tail's choices as partners, and the tail's rows gain the query
        rows (a row's k best can include earlier rows, whose scores only
        appear in earlier strips' columns). Selection is by score
        descending then index ascending — the tie rule `process.extract`
        documents — and deduplicates by partner index, since one pair can
        arrive from both directions with the identical score.

        The cutoff applies Python-side on the float64 strip scores; the
        boundary tolerance mirrors the all-pairs comparison and stays
        authoritative regardless of rapidfuzz version behaviour.
        """
        n = len(scored)
        if n < 2:
            return

        topn = self._topn
        cutoff: float = threshold - _BOUNDARY_TOLERANCE
        width = topn + 1  # state row width: k partners + self transiently

        max_block_memory = DEFAULT_BLOCK_MEMORY if self._max_block_memory is None else int(self._max_block_memory)
        block_rows = block_rows_for(n, probe_itemsize(scorer, dtype="float64"), max_block_memory)
        workers = resolve_workers(self._workers, distributed=self.distributed)

        state_score = np.full((n, width), -np.inf, dtype=np.float64)
        state_index = np.full((n, width), -1, dtype=np.int64)

        start = 0
        while start < n - 1:
            end = min(start + block_rows, n)
            strip = np.asarray(
                process.cdist(
                    scored[start:end],
                    scored[start + 1 :],
                    scorer=scorer,
                    dtype="float64",
                    workers=workers,
                )
            )

            q_idx, q_sco = _select_top(strip, start + 1, width)
            c_idx, c_sco = _select_top(strip.T, start, width)
            _merge_topk(
                state_score,
                state_index,
                np.concatenate(
                    [
                        np.repeat(np.arange(start, end), width),
                        np.repeat(np.arange(start + 1, start + 1 + strip.shape[1]), width),
                    ]
                ),
                np.concatenate([q_idx.ravel(), c_idx.ravel()]),
                np.concatenate([q_sco.ravel(), c_sco.ravel()]),
                width,
            )

            start = end

        seen: set[tuple[int, int]] = set()
        for pos in range(n):
            partners = 0
            for slot in range(width):
                if partners == topn:
                    break
                index = int(state_index[pos, slot])
                if index < 0:
                    break
                if index == pos:
                    continue
                partners += 1
                if state_score[pos, slot] < cutoff:
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
