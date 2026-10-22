"""Liken deduper unit tests"""

import math

import numpy as np
import pyarrow as pa

import liken as lk


def test_cosine_matches_a_pair_exactly_at_the_threshold():
    """Rows [1, 1] and [1, 0] have similarity `1/sqrt(2)`; a threshold equal to it must match."""
    threshold = float(1.0 / np.sqrt(2.0))

    pairs = list(lk.cosine(threshold=threshold)._gen_similarity_pairs(pa.table({"x": [1.0, 1.0], "y": [1.0, 0.0]})))

    assert pairs == [(0, 1)]


def test_cosine_does_not_match_below_the_threshold():
    threshold = float(math.nextafter(1.0 / np.sqrt(2.0), math.inf))

    pairs = list(lk.cosine(threshold=threshold)._gen_similarity_pairs(pa.table({"x": [1.0, 1.0], "y": [1.0, 0.0]})))

    assert pairs == []
