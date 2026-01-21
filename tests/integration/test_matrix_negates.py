"""Narrow integration tests for specific behaviour of individual stratgies"""

from __future__ import annotations

import typing

import pytest

from dupegrouper import Duped
from dupegrouper.constants import CANONICAL_ID
from dupegrouper.custom import register
from dupegrouper.strats_combinations import On
from dupegrouper.strats_library import (
    cosine,
    exact,
    fuzzy,
    jaccard,
    lsh,
    str_contains,
    str_endswith,
    str_startswith,
    tfidf,
)



# fmt: off

PARAMS = [
    #
    (str_startswith(pattern="a"), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 12]),
    (~str_startswith(pattern="a"), [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0]),
    #
    (str_endswith(pattern=".com"), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12]),
    (~str_endswith(pattern=".com"), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    #
    (str_contains(pattern="@example"), [0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 10, 11, 12]),
    (~str_contains(pattern="@example"), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9]),
]

# fmt: on


@pytest.mark.parametrize("strategy, expected_canonical_id", PARAMS)
def test_matrix_negates_inline(strategy, expected_canonical_id, dataframe, helpers):

    df, spark_session = dataframe

    dg = Duped(df, spark_session=spark_session)
    dg.apply(strategy)
    dg.canonicalize("email")

    assert helpers.get_column_as_list(dg.df, CANONICAL_ID) == expected_canonical_id

@pytest.mark.parametrize("strategy, expected_canonical_id", PARAMS)
def test_matrix_negates_dict(strategy, expected_canonical_id, dataframe, helpers):

    df, spark_session = dataframe

    dg = Duped(df, spark_session=spark_session)
    dg.apply({"email": strategy})
    dg.canonicalize()

    assert helpers.get_column_as_list(dg.df, CANONICAL_ID) == expected_canonical_id

@pytest.mark.parametrize("strategy, expected_canonical_id", PARAMS)
def test_matrix_negates_on(strategy, expected_canonical_id, dataframe, helpers):

    df, spark_session = dataframe

    dg = Duped(df, spark_session=spark_session)
    dg.apply(On("email", strategy))
    dg.canonicalize()

    assert helpers.get_column_as_list(dg.df, CANONICAL_ID) == expected_canonical_id
