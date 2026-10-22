"""Liken deduper unit tests"""

import math

import pyarrow as pa

import liken as lk


def test_jaccard_matches_a_pair_exactly_at_the_threshold():
    """Two rows sharing 2 of their 4 members have jaccard similarity exactly 0.5."""
    table = pa.table({"x": ["a", "a"], "y": ["b", "b"], "z": ["c", "d"]})

    pairs = list(lk.jaccard(threshold=0.5)._gen_similarity_pairs(table))

    assert pairs == [(0, 1)]


def test_jaccard_does_not_match_below_the_threshold():
    table = pa.table({"x": ["a", "a"], "y": ["b", "b"], "z": ["c", "d"]})

    pairs = list(lk.jaccard(threshold=math.nextafter(0.5, math.inf))._gen_similarity_pairs(table))

    assert pairs == []
