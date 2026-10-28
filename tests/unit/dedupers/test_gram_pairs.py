"""Unit tests for the shared Gram-strip pair generator."""

import math
import random

import numpy as np
import pytest

from liken.dedupers._blockwise import DEFAULT_BLOCK_MEMORY
from liken.dedupers._blockwise import gram_strip_pairs


def _reference_pairs(members, sizes, threshold, scoring):
    """Pure-Python reference: the per-row loop's pair output, in yield order.

    Mirrors the set-based comparisons of `jaccard` / `token_overlap`:
    a pair needs a non-empty intersection, and its score comes from
    (count, |A|, |B|) under the scoring formula.
    """
    sets = [set(m.tolist()) for m in members]
    n = len(sets)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            intersection = sets[i] & sets[j]
            if not intersection:
                continue
            denominator = len(sets[i] | sets[j]) if scoring == "jaccard" else min(len(sets[i]), len(sets[j]))
            if len(intersection) / denominator >= threshold:
                out.append((i, j))
    return out


def _seeded_members(n, *, seed=7, vocab=12, empty_rate=0.2):
    """Deterministic row member arrays, some rows empty."""
    rng = random.Random(seed)
    members = []
    for _ in range(n):
        if rng.random() < empty_rate:
            members.append(np.array([], dtype=np.int64))
            continue
        size = rng.randrange(1, 5)
        members.append(np.array(rng.sample(range(vocab), size), dtype=np.int64))
    return members


def _sizes_for(members):
    return np.array([len(m) for m in members], dtype=np.int64)


def _assert_matches_reference(members, threshold, scoring, **kwargs):
    expected = _reference_pairs(members, _sizes_for(members), threshold, scoring)
    got = list(gram_strip_pairs(members, _sizes_for(members), threshold, scoring, **kwargs))
    assert got == expected


##############################
# reference-loop equivalence #
#############################


@pytest.mark.parametrize("scoring", ["jaccard", "token_overlap"])
@pytest.mark.parametrize("threshold", [0.01, 0.5, 0.95])
def test_gram_strip_pairs_matches_the_reference_loop(scoring, threshold):
    _assert_matches_reference(_seeded_members(40), threshold, scoring)


@pytest.mark.parametrize("scoring", ["jaccard", "token_overlap"])
def test_gram_strip_pairs_wide_vocabulary_matches_the_reference_loop(scoring):
    """Distinct members per row: every pair shares at most one member."""
    _assert_matches_reference(_seeded_members(30, seed=11, vocab=200), 0.01, scoring)


def test_gram_strip_pairs_small_cap_matches_the_reference_loop():
    """A cap too small for one strip row clamps to one-row blocks."""
    _assert_matches_reference(_seeded_members(25), 0.5, "jaccard", max_block_memory=1)


def test_gram_strip_pairs_degenerate_cap_clamps_instead_of_raising():
    """A non-positive cap clamps to one-row blocks, never raises."""
    members = _seeded_members(20)
    _assert_matches_reference(members, 0.5, "jaccard", max_block_memory=0)
    _assert_matches_reference(members, 0.5, "token_overlap", max_block_memory=-8)


def test_gram_strip_pairs_none_cap_uses_the_library_default():
    members = _seeded_members(20)
    _assert_matches_reference(members, 0.5, "jaccard", max_block_memory=None)
    assert DEFAULT_BLOCK_MEMORY > 0


#####################
# output semantics #
####################


def test_gram_strip_pairs_empty_rows_never_match():
    """Rows with no members produce no pairs, in either direction."""
    members = [
        np.array([], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
    ]

    got = list(gram_strip_pairs(members, _sizes_for(members), 0.5, "jaccard"))

    assert got == [(1, 3)]


def test_gram_strip_pairs_threshold_is_inclusive():
    """Two rows sharing 2 of a 4-member union score jaccard exactly 0.5."""
    members = [
        np.array([0, 1], dtype=np.int64),
        np.array([0, 1, 2, 3], dtype=np.int64),
    ]
    sizes = _sizes_for(members)

    assert list(gram_strip_pairs(members, sizes, 0.5, "jaccard")) == [(0, 1)]
    assert list(gram_strip_pairs(members, sizes, math.nextafter(0.5, math.inf), "jaccard")) == []


def test_gram_strip_pairs_orders_pairs_i_then_j_ascending():
    """Yield order matches the replaced loop: i ascending, then j ascending."""
    members = _seeded_members(30, seed=3)

    got = list(gram_strip_pairs(members, _sizes_for(members), 0.01, "token_overlap"))

    assert got == sorted(got)


def test_gram_strip_pairs_interleaved_order_across_row_blocks():
    """Multi-block output is globally i-then-j, not block-by-block.

    With several row blocks, one i-block's pairs into a far j-block must
    still precede a later i-block's pairs into an earlier j-block: the
    loop's (i, j) order crosses block boundaries. A block-by-block
    emission breaks this; a large block_count with a small cap forces the
    multi-block path.
    """
    members = _seeded_members(60, seed=17, empty_rate=0.0)
    got = list(gram_strip_pairs(members, _sizes_for(members), 0.01, "jaccard", max_block_memory=4))

    assert got == sorted(got)
    assert got == _reference_pairs(members, _sizes_for(members), 0.01, "jaccard")


def test_gram_strip_pairs_fewer_than_two_rows_yields_nothing():
    assert list(gram_strip_pairs([], np.array([], dtype=np.int64), 0.5, "jaccard")) == []
    single = [np.array([0], dtype=np.int64)]
    assert list(gram_strip_pairs(single, _sizes_for(single), 0.5, "jaccard")) == []


def test_gram_strip_pairs_all_rows_empty_yields_nothing():
    members = [np.array([], dtype=np.int64) for _ in range(4)]

    assert list(gram_strip_pairs(members, _sizes_for(members), 0.01, "jaccard")) == []


##########################
# float32 exactness guard #
##########################


def test_gram_strip_pairs_raises_when_a_row_size_exceeds_the_guard():
    """A row size above 2^24 would break float32 count exactness."""
    members = [np.array([0], dtype=np.int64), np.array([0], dtype=np.int64)]
    sizes = np.array([2**24 + 1, 1], dtype=np.int64)

    with pytest.raises(ValueError, match=r"2\^24|16777216"):
        list(gram_strip_pairs(members, sizes, 0.5, "jaccard"))
