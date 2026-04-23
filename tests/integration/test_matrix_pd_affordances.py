"""Ensure pandas affordances produce identical results to core API."""

from __future__ import annotations

import pandas as pd

import liken as lk


DATA = [
    [1, "apple", 0, 0, 1, "sweden", "reddit"],
    [2, "apple", 0, 0.5, 1, "france", "facebook"],
    [3, "appl", 0, 0.5, 1, "spain", "reddit"],
    [4, "banana", 0.1, 0.5, 1.1, "uk", "reddit"],
    [5, "banan", 0, 0, 0, "spain", "youtube"],
    [6, "banana", 0, 0.1, 0, "spain", "reddit"],
]

COLS = ["id", "text", "num_1", "num_2", "num_3", "cat_1", "cat_2"]

df = pd.DataFrame(data=DATA, columns=COLS)


def test_affordance_fuzzy():

    COLS = "text"

    # Core API
    core = lk.dedupe(df).apply(lk.fuzzy(threshold=0.6)).drop_duplicates(COLS)

    # Pandas affordance API
    aff = df.fuzzy.drop_duplicates(COLS, threshold=0.6)

    assert_equal(core, aff)


def test_affordance_tfidf():

    COLS = "text"

    # Core API
    core = lk.dedupe(df).apply(lk.tfidf(threshold=0.6, topn=2)).drop_duplicates(COLS)

    # Pandas affordance API
    aff = df.tfidf.drop_duplicates(COLS, threshold=0.6, topn=2)

    assert_equal(core, aff)


def test_affordance_lsh():

    COLS = "text"

    # Core API
    core = lk.dedupe(df).apply(lk.lsh(threshold=0.6, ngram=2)).drop_duplicates(COLS)

    # Pandas affordance API
    aff = df.lsh.drop_duplicates(COLS, threshold=0.6, ngram=2)

    assert_equal(core, aff)


def test_affordance_jaccard():

    COLS = ("cat_1", "cat_2")

    # Core API
    core = lk.dedupe(df).apply(lk.jaccard(threshold=0.6)).drop_duplicates(COLS)

    # Pandas affordance API
    aff = df.jaccard.drop_duplicates(COLS, threshold=0.6)

    assert_equal(core, aff)


def test_affordance_cosine():

    COLS = ("num_1", "num_2", "num_3")

    # Core API
    core = lk.dedupe(df).apply(lk.cosine(threshold=0.6)).drop_duplicates(COLS)

    # Pandas affordance API
    aff = df.cosine.drop_duplicates(COLS, threshold=0.6)

    assert_equal(core, aff)


def assert_equal(core: pd.DataFrame, aff: pd.DataFrame):
    pd.testing.assert_frame_equal(
        core.sort_values("id").reset_index(drop=True),
        aff.sort_values("id").reset_index(drop=True),
    )
