"""Ensure pandas affordances produce identical results to core API."""

from __future__ import annotations

import pandas as pd
import pytest

import liken as lk


DATA = [
    [1, "apple"],
    [2, "apple"],
    [3, "appl"],
    [4, "banana"],
    [5, "banan"],
    [6, "banana"],
]

COLS = ["id", "text"]

df = pd.DataFrame(data=DATA, columns=COLS)


def test_affordance_fuzzy():

    # Core API
    core = (
        lk.dedupe(df)
        .apply(lk.fuzzy())
        .drop_duplicates("text")
    )

    # Pandas affordance API
    aff = df.lk.fuzzy().drop_duplicates(columns="text")

    assert_equal(core, aff)



def test_affordance_tfidf():

    # Core API
    core = (
        lk.dedupe(df)
        .apply(lk.tfidf())
        .drop_duplicates("text")
    )

    # Pandas affordance API
    aff = df.lk.tfidf().drop_duplicates(columns="text")

    assert_equal(core, aff)


def test_affordance_lsh():

    # Core API
    core = (
        lk.dedupe(df)
        .apply(lk.lsh())
        .drop_duplicates("text")
    )

    # Pandas affordance API
    aff = df.lk.lsh().drop_duplicates(columns="text")

    assert_equal(core, aff)

def assert_equal(core: pd.DataFrame, aff: pd.DataFrame):
    pd.testing.assert_frame_equal(
        core.sort_values("id").reset_index(drop=True),
        aff.sort_values("id").reset_index(drop=True),
    )