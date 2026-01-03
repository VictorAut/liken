"""
Full integration test for each backend wrapper and each strategy via
a cartesian product of (backend X strategy). For respective lower-level tests
of backend wrappers and deduplication strategies, please see unit tests.
"""

from __future__ import annotations

import pytest

from dupegrouper import DupeGrouper
from dupegrouper.definitions import CANONICAL_ID
from dupegrouper.strategies import strategies


STRATEGY_CLASSES = (
    strategies.Exact,
    strategies.Fuzzy,
    strategies.TfIdf,
)

STRATEGY_PARAMS: tuple[dict, ...] = (
    {},  # for exact
    {"threshold": 0.55},  # for fuzzy
    {"ngram": (1, 1), "threshold": 0.80, "topn": 2},  # for tfidf
)

EXPECTED_CANONICAL_ID = (
    [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 11, 12, 13],  # for exact
    [1, 2, 3, 3, 2, 6, 3, 2, 1, 1, 2, 12, 13],  # for fuzzy
    [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 12, 13],  # for tfidf
)


@pytest.mark.parametrize(
    "strategy_class, strategy_params, expected_canonical_id",
    zip(STRATEGY_CLASSES, STRATEGY_PARAMS, EXPECTED_CANONICAL_ID),
    ids=[cls.__name__ for cls in STRATEGY_CLASSES],
)
def test_dedupe_matrix(strategy_class, strategy_params, expected_canonical_id, dataframe, helpers):

    df, spark_session, id = dataframe

    dg = DupeGrouper(df=df, spark_session=spark_session, id=id)

    # single strategy item addition
    dg.add_strategy(strategy_class(**strategy_params))
    dg.dedupe("address")
    df1 = dg.df

    # dictionary straegy addition
    dg.add_strategy({"address": [strategy_class(**strategy_params)]})
    dg.dedupe()
    df2 = dg.df

    assert helpers.get_column_as_list(df1, CANONICAL_ID) == expected_canonical_id == helpers.get_column_as_list(df2, CANONICAL_ID)
