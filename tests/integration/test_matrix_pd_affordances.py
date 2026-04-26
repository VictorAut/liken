"""Ensure pandas affordances produce identical results to core API."""

from __future__ import annotations

import pandas as pd
import pytest

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

PARAMS = [
    ("text", "fuzzy", {"threshold":0.6}),
    ("text", "tfidf", {"threshold":0.6, "topn":2}),
    ("text", "lsh", {"threshold":0.6, "ngram":2}),
    (("cat_1", "cat_2"), "jaccard", {"threshold":0.6}),
    (("num_1", "num_2", "num_3"), "cosine", {"threshold":0.6}),
]

IDS = [i[1] for i in PARAMS]

@pytest.mark.parametrize(
    "columns, deduper, kwargs",
    PARAMS,
    ids=IDS,
)
def test_pd_affordance(columns, deduper, kwargs, request):

    backend = request.config.getoption("--backend")

    if backend != "pandas":
        pytest.skip("Only a pandas test")

    # Core API
    core = lk.dedupe(df).apply(getattr(lk, deduper)(**kwargs)).drop_duplicates(columns)

    # Pandas affordance API
    aff = getattr(df, deduper).drop_duplicates(columns, **kwargs)

    assert_equal(core, aff)


def assert_equal(core: pd.DataFrame, aff: pd.DataFrame):
    pd.testing.assert_frame_equal(
        core.sort_values("id").reset_index(drop=True),
        aff.sort_values("id").reset_index(drop=True),
    )
