"""Block-wise batched pair generation shared by the cdist dedupers.

Scores pairs in horizontal strips of `block_rows` queries against the
remaining tail, one `process.cdist` call per strip, and filters pairs with
a vectorized mask per block. Memory is bounded by a block cap; small
inputs degenerate into a single batched call.

The set dedupers (`jaccard`, `token_overlap`) share `gram_strip_pairs`,
which scores pairs from blocked one-hot Gram strips of member sets under
the same memory cap. `inverted_index_pairs` is the high-vocabulary
alternative: it emits one key per shared member and counts pairs through
the inverted index.

Pair output is byte-identical to the per-row loops it replaces, yield
order included.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING
from typing import Final
from typing import Literal

import numpy as np
from rapidfuzz import process


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

    import numpy.typing as npt

    from liken.types import SimilarPairIndices


# Library default for the per-block memory cap, in bytes. The WI013
# benchmark study (2026-10-27) confirmed it produces sensible block sizes.
DEFAULT_BLOCK_MEMORY: Final[int] = 64 * 1024 * 1024

# float32 sgemm is exact for intersection counts up to 2^24. A count beyond
# that would need a single row to hold over 16M distinct members, which no
# real compound column reaches, so the guard is a cheap shape check, not a
# dtype change.
_MAX_FLOAT32_EXACT_COUNT: Final[int] = 2**24


def block_rows_for(n: int, itemsize: int, cap_bytes: int) -> int:
    """Rows per horizontal strip under a block memory cap.

    The cap bounds the score matrix of one `cdist` call: a strip of
    `block_rows` queries against the remaining `n - 1` choices costs
    `block_rows * (n - 1) * itemsize` bytes. Degenerate inputs — a cap
    below one row's score band, a zero `itemsize` or `n` — clamp to a
    one-row strip. Peak memory may then exceed the cap by that band.

    Args:
        n: Number of present (non-missing) values, upper-bounding the
            choice count of any strip.
        itemsize: Byte size of one score, from the score dtype.
        cap_bytes: Memory cap for one block's score matrix.

    Returns:
        The strip row count, at least 1.
    """
    denominator = n * itemsize
    if denominator <= 0 or cap_bytes <= 0:
        return 1
    return max(1, cap_bytes // denominator)


def probe_itemsize(compare: Callable, *, dtype: str | None) -> int:
    """Byte size of one score, from one tiny `cdist` call.

    The score dtype is the scorer's; probing it once per generation keeps
    the block-size formula in bytes without hard-coding per scorer. The
    probe must run with the same `dtype` as the real block calls, or the
    resolved strip size mis-sizes the matrix.

    Args:
        compare: The scorer the block loop will run.
        dtype: The `dtype` the block calls will pass to `cdist`.

    Returns:
        The itemsize of the score dtype, in bytes.
    """
    matrix = process.cdist(["abc"], ["abd", "abe"], scorer=compare, dtype=dtype)
    return int(matrix.dtype.itemsize)


def resolve_workers(workers: int | None, *, distributed: bool) -> int:
    """Resolve the `workers` knob to a `cdist` worker count.

    `None` means "choose automatically": all cores locally, single-threaded
    on a distributed backend, where the executor owns parallelism and a
    partition task's all-cores request would oversubscribe the cores. An
    explicit integer always wins, `-1` included. `workers` is expected
    validated by the constructing deduper; this function does not re-check.

    Args:
        workers: The validated knob value, or `None` for automatic.
        distributed: Whether the deduper runs under a distributed backend.

    Returns:
        The worker count for `process.cdist`.
    """
    if workers is None:
        return 1 if distributed else (os.cpu_count() or 1)
    return workers


def similar_pairs_blockwise(
    scored: list,
    *,
    compare: Callable,
    low: bool,
    bound: float,
    block_rows: int,
    workers: int,
    dtype: str | None = None,
) -> Iterator[SimilarPairIndices]:
    """Yield matching index pairs in block-wise batched shape.

    Horizontal strips of `block_rows` queries are scored against the
    remaining tail in one `cdist` call each, and pairs are extracted with
    a vectorized mask. In a block rectangle starting at `start`, the
    diagonal element (r, r) compares `scored[start + r]` against
    `scored[start + r + 1]`: a real pair, not a self-pair, so the mask is
    `triu(k=0)`.

    Args:
        scored: The non-missing values, in row order.
        compare: A rapidfuzz scorer or compatible callable receiving
            `(query, choice, score_cutoff=None)`.
        low: Whether pairs match at or below `bound` (distance metrics)
            instead of at or above it (similarity scorers). `bound` must
            already carry the deduper's boundary tolerance.
        bound: The comparison bound in the scorer's own scale.
        block_rows: Rows per strip, at least 1.
        workers: Worker count passed to `process.cdist`.
        dtype: The score dtype passed to `process.cdist`, or `None` for
            the scorer's default. Must match the `dtype` the block size
            was probed with, and the dtype the replaced loop scored in.

    Yields:
        Pairs of indices into `scored`, matching the replaced per-row
        loops' output and yield order.
    """
    n = len(scored)
    if n < 2 or block_rows < 1:
        return

    start = 0
    while start < n - 1:
        end = min(start + block_rows, n)
        choices = scored[start + 1 :]

        matrix = process.cdist(
            scored[start:end],
            choices,
            scorer=compare,
            dtype=dtype,
            workers=workers,
        )

        scores: np.ndarray = np.asarray(matrix)
        mask = scores >= bound if not low else scores <= bound
        mask &= np.triu(np.ones(mask.shape, dtype=bool), k=0)

        for r, c in np.argwhere(mask):
            yield start + int(r), start + 1 + int(c)

        start = end


def strip_rows_for(v: int, cap_bytes: int) -> int:
    """Rows per Gram strip under a block memory cap.

    The cap bounds one row-block strip: a strip of `strip_rows` rows holds
    `strip_rows * v` float32 indicator entries, so its size is
    `strip_rows * v * 4` bytes. Degenerate inputs — a cap below one row's
    strip, a zero `v` or cap — clamp to a one-row strip. Peak memory may
    then exceed the cap by that strip; a single row strip is `v * 4` bytes
    and cannot be blocked further.

    Args:
        v: Vocabulary size, the strip's column count.
        cap_bytes: Memory cap for one row-block strip.

    Returns:
        The strip row count, at least 1.
    """
    denominator = v * 4
    if denominator <= 0 or cap_bytes <= 0:
        return 1
    return max(1, cap_bytes // denominator)


def _gram_strip_block_rows(v: int, cap_bytes: int) -> int:
    """Strip rows also fitting the score block under the memory cap.

    A square `m x m'` score block of float32 counts costs `m * m' * 4`
    bytes; with square blocks `m = m'`, so `m` is additionally bounded by
    the square root of `cap_bytes // 4`. The strip bound still applies:
    blocking the score block cannot shrink one row's `v * 4` strip.

    Args:
        v: Vocabulary size, the strip's column count.
        cap_bytes: Memory cap in bytes.

    Returns:
        The block row count, at least 1.
    """
    square_rows = max(1, math.isqrt(max(1, cap_bytes // 4)))
    return min(strip_rows_for(v, cap_bytes), square_rows)


def gram_strip_pairs(
    members: list[npt.NDArray[np.int64]],
    sizes: npt.NDArray[np.int64],
    threshold: float,
    scoring: Literal["jaccard", "token_overlap"],
    max_block_memory: int | None = None,
) -> Iterator[SimilarPairIndices]:
    """Yield matching row pairs from member sets via blocked Gram strips.

    Each row's members are one-hot encoded into a float32 strip (a row
    block of the full n x v indicator matrix); the product of two strips
    gives exact intersection counts for every row pair in the block
    rectangle, because a one-hot product counts shared members exactly.
    The full n x v indicator is never materialised: strips are built per
    row block, so peak memory is the strip plus one score block.

    Scores are computed from (count, |A|, |B|) per pair. "jaccard"
    divides the count by |A| + |B| - count; "token_overlap" divides it by
    min(|A|, |B|). The comparison is inclusive: a pair whose score is
    exactly equal to `threshold` matches.

    Args:
        members: Per-row member arrays holding dense member IDs
            (`0..v-1`). Rows with no members produce no pairs.
        sizes: Row member counts, `len(members[i])` per row.
        threshold: The inclusive score bound.
        scoring: The score formula, "jaccard" or "token_overlap".
        max_block_memory: Memory cap in bytes for one strip and one score
            block. `None` uses `DEFAULT_BLOCK_MEMORY`. Values too small
            for one row's strip clamp to one-row blocks. The cap bounds
            the strip and score block; the per-i-block pair buffer used to
            restore the loop's global order is a third peak term not
            bounded by the cap (proportional to the matches in one
            i-block).

    Yields:
        Pairs `(i, j)` with `i < j`, `i` ascending then `j` ascending,
        the order of the per-row loops this generator replaces.

    Raises:
        ValueError: If any row size exceeds 2^24, where float32 counts
            could no longer be exact.
    """
    n = len(members)
    if n < 2:
        return

    cap = DEFAULT_BLOCK_MEMORY if max_block_memory is None else int(max_block_memory)
    if int(sizes.max(initial=0)) > _MAX_FLOAT32_EXACT_COUNT:
        raise ValueError(
            f"a row holds more than the maximum of {_MAX_FLOAT32_EXACT_COUNT} distinct members;"
            " intersection counts would not be exact in float32"
        )

    # Distinct member IDs actually present, so v tracks the data, not the
    # nominal vocabulary passed in.
    v = 1 + max((int(m.max(initial=-1)) for m in members if len(m)), default=-1)

    if v <= 0:
        return

    block_rows = _gram_strip_block_rows(v, cap)

    def build_strip(start: int, end: int) -> np.ndarray:
        """One float32 row-block strip: `m x v` one-hot rows."""
        strip = np.zeros((end - start, v), dtype=np.float32)
        for local_row, member_array in enumerate(members[start:end]):
            strip[local_row, member_array] = 1.0
        return strip

    start_i = 0
    while start_i < n - 1:
        end_i = min(start_i + block_rows, n)
        strip_i = build_strip(start_i, end_i)
        size_i = sizes[start_i:end_i].astype(np.float64)[:, None]

        block_pairs: list[np.ndarray] = []
        block_row_starts: list[int] = []
        start_j = start_i
        while start_j < n:
            end_j = min(start_j + block_rows, n)
            strip_j = strip_i if start_j == start_i else build_strip(start_j, end_j)
            size_j = sizes[start_j:end_j].astype(np.float64)[None, :]

            counts: np.ndarray = (strip_i @ strip_j.T).astype(np.float64)
            if start_j == start_i:
                # The diagonal block holds every (i, j) ordering; the loop
                # emits only j > i, and zeroing the lower triangle and
                # diagonal before the dense pass also skips its
                # empty-intersection work.
                counts = np.triu(counts, k=1)

            # Dense, vectorised scoring over the whole block rectangle:
            # every (i, j) in the rectangle is scored once.
            size_ij = size_i + size_j
            if scoring == "jaccard":
                scores = np.divide(counts, size_ij - counts, out=np.zeros_like(counts), where=counts > 0)
            else:
                scores = np.divide(
                    counts,
                    np.minimum(size_i, size_j),
                    out=np.zeros_like(counts),
                    where=counts > 0,
                )

            # The loop skips pairs with an empty intersection, and a pair
            # can only have scored through a shared member, so the match
            # condition is score >= threshold over non-zero counts.
            np.greater_equal(scores, threshold, out=scores)

            # Buffer this rectangle's pairs. One i-block produces pairs from
            # several j-blocks, and the loop's i-then-j order interleaves
            # them (a pair into a far j-block precedes a pair into the next
            # j-block whose i is larger), so per-rectangle emission cannot
            # be global. Sorting the i-block's pairs lexicographically
            # restores the loop's exact order.
            block_pairs.append(np.argwhere(scores))
            block_row_starts.append(start_j)

            start_j = end_j

        if block_pairs:
            li_all = np.concatenate([bp[:, 0] for bp in block_pairs])
            lj_all = np.concatenate([bp[:, 1] for bp in block_pairs])
            gi_all = li_all + start_i
            # Each rectangle's pairs carry its own row offset; expand the
            # per-rectangle offsets to per-element before adding.
            gj_all = lj_all + np.repeat(block_row_starts, [len(bp) for bp in block_pairs])
            # lexsort is stable: last key is primary, so sort by j within i.
            order = np.lexsort((gj_all, gi_all))
            for gi, gj in zip(gi_all[order].tolist(), gj_all[order].tolist(), strict=True):
                yield int(gi), int(gj)

        start_i = end_i


def inverted_index_pairs(
    members: list[npt.NDArray[np.int64]],
    sizes: npt.NDArray[np.int64],
    threshold: float,
    scoring: Literal["jaccard", "token_overlap"],
    max_block_memory: int | None = None,
    *,
    max_chunk_bytes: int | None = None,
) -> Iterator[SimilarPairIndices]:
    """Yield matching row pairs from member sets via inverted-index counting.

    Every pair sharing at least one member emits one key per shared
    member; `np.unique` over the keys gives each pair's exact intersection
    count. Scores come from (count, |A|, |B|) under the same formulas and
    float64 boundary rules as `gram_strip_pairs`, and the comparison is
    inclusive.

    Peak key memory is bounded by the cap: groups are emitted in anchor
    slices and deduplicated per slice, so the full pre-dedup key array is
    never materialised. This is the regime for high-vocabulary data, where
    most values are rare and the Gram product of `gram_strip_pairs` would
    be mostly waste.

    Args:
        members: Per-row member arrays holding dense member IDs
            (`0..v-1`). Rows with no members produce no pairs.
        sizes: Row member counts, `len(members[i])` per row.
        threshold: The inclusive score bound.
        scoring: The score formula, "jaccard" or "token_overlap".
        max_block_memory: Memory cap in bytes bounding one in-flight key
            chunk. `None` uses `DEFAULT_BLOCK_MEMORY`. Values too small
            for one key clamp to one-key chunks.
        max_chunk_bytes: Harness-only override of the chunk cap, for
            forcing multi-chunk runs; keyword-only.

    Yields:
        Pairs `(i, j)` with `i < j`, `i` ascending then `j` ascending,
        the order of the per-row loops this generator replaces.
    """
    n = len(members)
    if n < 2:
        return

    # Flatten to (member, row) pairs. Rows arrive with their members
    # already dense-ID encoded and option-C filtered by the caller.
    lengths = np.fromiter((len(m) for m in members), dtype=np.int64, count=n)
    total = int(lengths.sum())
    if total == 0:
        return
    row_ids = np.repeat(np.arange(n, dtype=np.int64), lengths)
    member_ids = np.concatenate([np.asarray(m, dtype=np.int64) for m in members])

    # Sort by (member, row) so each value's rows are contiguous and
    # ascending: within a group, row i < row j implies rows[i] < rows[j].
    order = np.lexsort((row_ids, member_ids))
    rows_sorted = row_ids[order]
    members_sorted = member_ids[order]

    # Group boundaries: where the member ID changes.
    boundaries = np.flatnonzero(np.diff(members_sorted)) + 1
    group_starts = np.concatenate(([0], boundaries))
    group_ends = np.concatenate((boundaries, [total]))

    # Emit each group's C(k, 2) keys in anchor slices bounded by the chunk
    # cap. Key `lo * n + hi` is unique per pair and sorts lexicographic in
    # (lo, hi), which is the loop's i-then-j order. A slice holds one
    # anchor block's pairs: anchor a contributes its k - 1 - a later rows,
    # so a block of `block` anchors holds at most `block * (k - 1)` keys.
    cap = DEFAULT_BLOCK_MEMORY if max_block_memory is None else int(max_block_memory)
    chunk_cap = cap if max_chunk_bytes is None else int(max_chunk_bytes)
    budget = max(1, chunk_cap // 8)

    chunk_parts: list[np.ndarray] = []
    uniq_parts: list[np.ndarray] = []
    count_parts: list[np.ndarray] = []
    buffered = 0

    def flush() -> None:
        """Deduplicate the buffered keys into one (unique, counts) part."""
        nonlocal buffered, chunk_parts
        if not chunk_parts:
            return
        keys = np.concatenate(chunk_parts)
        uniq, counts = np.unique(keys, return_counts=True)
        uniq_parts.append(uniq)
        count_parts.append(counts)
        chunk_parts = []
        buffered = 0

    for start, end in zip(group_starts.tolist(), group_ends.tolist(), strict=True):
        k = end - start
        if k < 2:
            continue
        group_rows = rows_sorted[start:end]
        block = max(1, budget // (k - 1))
        for anchor_start in range(0, k - 1, block):
            anchor_end = min(anchor_start + block, k - 1)
            anchors = np.arange(anchor_start, anchor_end)
            run_lengths = (k - 1) - anchors
            run_total = int(run_lengths.sum())
            run_starts = np.cumsum(run_lengths) - run_lengths
            # Position within each anchor's run of later rows.
            positions = np.arange(run_total, dtype=np.int64) - np.repeat(run_starts, run_lengths)
            hi_idx = np.repeat(anchors + 1, run_lengths) + positions
            # Rows ascend within the group, so the anchor row is the
            # pair's lo without a min/max pass.
            keys = np.repeat(group_rows[anchors], run_lengths) * n + group_rows[hi_idx]
            chunk_parts.append(keys)
            buffered += keys.size
            if buffered >= budget:
                flush()
    flush()

    if not uniq_parts:
        return

    # Merge the per-chunk (unique, counts) parts into global unique keys
    # and exact intersection counts. Each part is already sorted-unique,
    # so the merge is a searchsplit into the globally sorted keyspace.
    if len(uniq_parts) == 1:
        uniq, counts = uniq_parts[0], count_parts[0]
    else:
        uniq = np.unique(np.concatenate(uniq_parts))
        counts = np.zeros(uniq.size, dtype=np.int64)
        for part_uniq, part_counts in zip(uniq_parts, count_parts, strict=True):
            counts[np.searchsorted(uniq, part_uniq)] += part_counts

    lo = uniq // n
    hi = uniq % n
    size_lo = sizes[lo]
    size_hi = sizes[hi]
    denominator = size_lo + size_hi - counts if scoring == "jaccard" else np.minimum(size_lo, size_hi)
    # int64 / int64 true-divides to float64, matching the loop's arithmetic.
    scores = counts / denominator
    matches = scores >= threshold
    for lo_val, hi_val in zip(lo[matches].tolist(), hi[matches].tolist(), strict=True):
        yield int(lo_val), int(hi_val)
