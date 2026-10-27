"""Block-wise batched pair generation shared by the cdist dedupers.

Scores pairs in horizontal strips of `block_rows` queries against the
remaining tail, one `process.cdist` call per strip, and filters pairs with
a vectorized mask per block. Memory is bounded by a block cap; small
inputs degenerate into a single batched call.

Pair output is byte-identical to the per-row loops it replaces, yield
order included.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from typing import Final

import numpy as np
from rapidfuzz import process


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

    from liken.types import SimilarPairIndices


# Library default for the per-block memory cap, in bytes. The WI013
# benchmark study (2026-10-27) confirmed it produces sensible block sizes.
DEFAULT_BLOCK_MEMORY: Final[int] = 64 * 1024 * 1024


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


def resolve_workers(workers: int | None, *, distributed: bool) -> int:
    """Resolve the `workers` knob to a `cdist` worker count.

    `None` means "choose automatically": all cores locally, single-threaded
    on a distributed backend, where the executor owns parallelism and a
    partition task's all-cores request would oversubscribe the cores. An
    explicit integer always wins, `-1` included.

    Args:
        workers: The user's knob value, or `None` for automatic.
        distributed: Whether the deduper runs under a distributed backend.

    Returns:
        The worker count for `process.cdist`.

    Raises:
        InvalidArgumentValueError: if `workers` is an integer below -1.
    """
    from liken.exceptions import InvalidArgumentValueError

    if workers is None:
        return 1 if distributed else (os.cpu_count() or 1)
    if workers == 0 or workers < -1:
        raise InvalidArgumentValueError(f"`workers` must be a nonzero integer or -1, got {workers!r}")
    return workers


def similar_pairs_blockwise(
    scored: list,
    *,
    compare: Callable,
    low: bool,
    bound: float,
    block_rows: int,
    workers: int,
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

    Yields:
        Pairs of indices into `scored`, matching the replaced per-row
        loops' output and yield order.
    """
    n = len(scored)
    if n < 2 or block_rows < 1:
        return

    kwargs: dict = {"workers": workers}

    start = 0
    while start < n - 1:
        end = min(start + block_rows, n)
        choices = scored[start + 1 :]

        matrix = process.cdist(
            scored[start:end],
            choices,
            scorer=compare,
            **kwargs,
        )

        scores: np.ndarray = np.asarray(matrix)
        mask = scores >= bound if not low else scores <= bound
        mask &= np.triu(np.ones(mask.shape, dtype=bool), k=0)

        for r, c in np.argwhere(mask):
            yield start + int(r), start + 1 + int(c)

        start = end
