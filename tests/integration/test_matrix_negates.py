"""Narrow integration tests for specific behaviour of individual stratgies"""

from __future__ import annotations

import pytest

from liken import Dedupe
from liken._constants import CANONICAL_ID
from liken.rules import Rules
from liken.rules import isna
from liken.rules import on
from liken.rules import str_contains
from liken.rules import str_endswith
from liken.rules import str_len
from liken.rules import str_startswith


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

    lk = Dedupe(df, spark_session=spark_session)
    lk.apply(Rules(on(col, strategy)))
    df = lk.canonicalize()

    assert helpers.get_column_as_list(df, CANONICAL_ID) == expected_canonical_id
