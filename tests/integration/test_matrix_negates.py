"""Narrow integration tests for specific behaviour of individual stratgies"""

from __future__ import annotations


import pytest

from dupegrouper import Dedupe
from enlace._constants import CANONICAL_ID
from enlace.rules import Rules, on, str_contains, str_endswith, str_startswith, isna, str_len


# fmt: off

PARAMS = [
    #
    (str_len(min_len=15, max_len=22), "email", [0, 1, 2, 0, 4, 5, 0, 0, 8, 9]),
    (~str_len(min_len=15, max_len=22), "email", [0, 1, 1, 3, 1, 1, 6, 7, 1, 1]),
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

    dp = Dedupe(df, spark_session=spark_session)
    dp.apply(Rules(on(col, strategy)))
    df = dp.canonicalize()

    assert helpers.get_column_as_list(df, CANONICAL_ID) == expected_canonical_id
