"""Narrow integration tests for specific behaviour of individual stratgies"""

from __future__ import annotations


import pytest

from dupegrouper import Dedupe
from dupegrouper._constants import CANONICAL_ID
from dupegrouper.rules import Rules, on, str_contains, str_endswith, str_startswith, isna


# fmt: off

PARAMS = [
    #
    (str_startswith(pattern="a"), "email", [0, 1, 1, 3, 4, 5, 6, 7, 8, 9]),
    (~str_startswith(pattern="a"), "email", [0, 1, 2, 0, 0, 0, 0, 0, 0, 0]),
    #
    (str_endswith(pattern=".com"), "email", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (~str_endswith(pattern=".com"), "email", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    #
    (str_contains(pattern="@example"), "email", [0, 1, 0, 0, 0, 0, 0, 0, 8, 0]),
    (~str_contains(pattern="@example"), "email", [0, 1, 2, 3, 4, 5, 6, 7, 1, 9]),
    #
    (isna(), "address", [0, 1, 2, 3, 4, 5, 6, 7, 4, 9]),
    (~isna(), "address", [0, 0, 0, 0, 4, 0, 0, 0, 8, 0]),
]

# fmt: on


# Negation is strongly encouraged to be only for the Rules API!


@pytest.mark.parametrize("strategy, col, expected_canonical_id", PARAMS)
def test_matrix_negates(strategy, col, expected_canonical_id, dataframe, helpers):

    df, spark_session = dataframe

    dg = Dedupe(df, spark_session=spark_session)
    dg.apply(Rules(on(col, strategy)))
    dg.canonicalize()

    assert helpers.get_column_as_list(dg.df, CANONICAL_ID) == expected_canonical_id
