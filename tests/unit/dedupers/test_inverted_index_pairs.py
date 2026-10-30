"""Unit tests for the inverted-index pair generator (WI069 V2)."""

import math
import random

import numpy as np
import pytest

from liken.dedupers._blockwise import inverted_index_pairs


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


def _seeded_members(n, *, seed=7, vocab=12, empty_rate=0.2, max_size=5):
    """Deterministic row member arrays, some rows empty."""
    rng = random.Random(seed)
    members = []
    for _ in range(n):
        if rng.random() < empty_rate:
            members.append(np.array([], dtype=np.int64))
            continue
        size = rng.randrange(1, max_size)
        members.append(np.array(rng.sample(range(vocab), size), dtype=np.int64))
    return members


def _sizes_for(members):
    return np.array([len(m) for m in members], dtype=np.int64)


def _assert_matches_reference(members, threshold, scoring, **kwargs):
    expected = _reference_pairs(members, _sizes_for(members), threshold, scoring)
    got = list(inverted_index_pairs(members, _sizes_for(members), threshold, scoring, **kwargs))
    assert got == expected


##############################
# reference-loop equivalence #
#############################


@pytest.mark.parametrize("scoring", ["jaccard", "token_overlap"])
@pytest.mark.parametrize("threshold", [0.01, 0.5, 0.95])
def test_inverted_index_pairs_matches_the_reference_loop(scoring, threshold):
    _assert_matches_reference(_seeded_members(40), threshold, scoring)


@pytest.mark.parametrize("scoring", ["jaccard", "token_overlap"])
def test_inverted_index_pairs_wide_vocabulary_matches_the_reference_loop(scoring):
    """High-vocabulary data: every pair shares at most one member."""
    _assert_matches_reference(_seeded_members(30, seed=11, vocab=200), 0.01, scoring)


@pytest.mark.parametrize("scoring", ["jaccard", "token_overlap"])
def test_inverted_index_pairs_single_shared_value(scoring):
    """One value held by every row: the densest possible inverted index."""
    members = [np.array([0, i], dtype=np.int64) for i in range(1, 8)]
    _assert_matches_reference(members, 0.3, scoring)


#####################
# chunked emission #
####################


def test_inverted_index_pairs_small_cap_matches_the_unchunked_run():
    """A cap forcing multi-chunk emission matches the unchunked output."""
    members = _seeded_members(40, seed=19, empty_rate=0.0)
    sizes = _sizes_for(members)
    unchunked = list(inverted_index_pairs(members, sizes, 0.01, "jaccard"))
    chunked = list(inverted_index_pairs(members, sizes, 0.01, "jaccard", max_chunk_bytes=256))
    assert chunked == unchunked


def test_inverted_index_pairs_single_group_larger_than_one_chunk():
    """One value shared by many rows is chunked across emission slices."""
    members = [np.array([0, 1], dtype=np.int64) for _ in range(30)]
    sizes = _sizes_for(members)
    chunked = list(inverted_index_pairs(members, sizes, 1.0, "jaccard", max_chunk_bytes=64))
    expected = [(i, j) for i in range(30) for j in range(i + 1, 30)]
    assert chunked == expected


def test_inverted_index_pairs_global_order_holds_across_chunks():
    """Yield order is i-then-j ascending even when emission is chunked."""
    members = _seeded_members(60, seed=17, empty_rate=0.0)
    got = list(inverted_index_pairs(members, _sizes_for(members), 0.01, "jaccard", max_chunk_bytes=128))
    assert got == sorted(got)
    assert got == _reference_pairs(members, _sizes_for(members), 0.01, "jaccard")


#####################
# output semantics #
####################


def test_inverted_index_pairs_empty_rows_never_match():
    """Rows with no members produce no pairs, in either direction."""
    members = [
        np.array([], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
    ]

    got = list(inverted_index_pairs(members, _sizes_for(members), 0.5, "jaccard"))

    assert got == [(1, 3)]


def test_inverted_index_pairs_threshold_is_inclusive():
    """Two rows sharing 2 of a 4-member union score jaccard exactly 0.5."""
    members = [
        np.array([0, 1], dtype=np.int64),
        np.array([0, 1, 2, 3], dtype=np.int64),
    ]
    sizes = _sizes_for(members)

    assert list(inverted_index_pairs(members, sizes, 0.5, "jaccard")) == [(0, 1)]
    assert list(inverted_index_pairs(members, sizes, math.nextafter(0.5, math.inf), "jaccard")) == []


def test_inverted_index_pairs_orders_pairs_i_then_j_ascending():
    """Yield order matches the replaced loop: i ascending, then j ascending."""
    members = _seeded_members(30, seed=3)

    got = list(inverted_index_pairs(members, _sizes_for(members), 0.01, "token_overlap"))

    assert got == sorted(got)


def test_inverted_index_pairs_fewer_than_two_rows_yields_nothing():
    assert list(inverted_index_pairs([], np.array([], dtype=np.int64), 0.5, "jaccard")) == []
    single = [np.array([0], dtype=np.int64)]
    assert list(inverted_index_pairs(single, _sizes_for(single), 0.5, "jaccard")) == []


def test_inverted_index_pairs_all_rows_empty_yields_nothing():
    members = [np.array([], dtype=np.int64) for _ in range(4)]

    assert list(inverted_index_pairs(members, _sizes_for(members), 0.01, "jaccard")) == []


def test_inverted_index_pairs_no_shared_members_yields_nothing():
    """An inverted index with no repeated value emits no keys."""
    members = [np.array([i], dtype=np.int64) for i in range(6)]

    assert list(inverted_index_pairs(members, _sizes_for(members), 0.01, "jaccard")) == []


#####################
# float32 non-guard #
####################


def test_inverted_index_pairs_oversized_sizes_raise_nothing():
    """V2 counts are int64; the 2^24 float32-exactness bound is V1-only."""
    members = [np.array([0], dtype=np.int64), np.array([0], dtype=np.int64)]
    sizes = np.array([2**24 + 1, 1], dtype=np.int64)

    got = list(inverted_index_pairs(members, sizes, 0.5, "token_overlap"))
    assert got == [(0, 1)]
