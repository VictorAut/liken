"""
Full integration test for each backend wrapper and each strategy via
a cartesian product of (backend X strategy). For respective lower-level tests
of backend wrappers and deduplication strategies, please see unit tests.
"""

from __future__ import annotations

import pytest

from dupegrouper import DupeGrouper
from dupegrouper.definitions import GROUP_ID
from dupegrouper.strategies import exact, fuzzy, tfidf


STRATEGY_CLASSES = (
    exact.Exact,
    fuzzy.Fuzzy,
    tfidf.TfIdf,
)

STRATEGY_PARAMS: tuple[dict, ...] = (
    {},  # for exact
    {"tolerance": 0.45},  # for fuzzy
    {"ngram": (1, 1), "tolerance": 0.20, "topn": 2},  # for tfidf
)

EXPECTED_GROUP_ID = (
    [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 11, 12, 13],  # for exact
    [1, 2, 3, 3, 5, 6, 3, 2, 1, 1, 11, 12, 13],  # for fuzzy
    [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 12, 13],  # for tfidf
)


@pytest.mark.parametrize(
    "strategy_class, strategy_params, expected_group_id",
    zip(STRATEGY_CLASSES, STRATEGY_PARAMS, EXPECTED_GROUP_ID),
    ids=[cls.__name__ for cls in STRATEGY_CLASSES],
)
def test_dedupe_matrix(strategy_class, strategy_params, expected_group_id, dataframe, helpers):

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

    assert helpers.get_column_as_list(df1, GROUP_ID) == expected_group_id == helpers.get_column_as_list(df2, GROUP_ID)
