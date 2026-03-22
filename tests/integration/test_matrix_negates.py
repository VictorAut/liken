"""Narrow integration tests for specific behaviour of individual stratgies"""

from __future__ import annotations

import pytest

import liken as lk
from liken._constants import CANONICAL_ID


# fmt: off

PARAMS = [
    #
    (lk.rules.on("email").str_len(min_len=15, max_len=22), [0, 1, 2, 0, 4, 5, 0, 0, 8, 9]),
    (~lk.rules.on("email").str_len(min_len=15, max_len=22), [0, 1, 1, 3, 1, 1, 6, 7, 1, 1]),
    #
    (lk.rules.on("email").str_startswith(pattern="a"), [0, 1, 1, 3, 4, 5, 6, 7, 8, 9]),
    (~lk.rules.on("email").str_startswith(pattern="a"), [0, 1, 2, 0, 0, 0, 0, 0, 0, 0]),
    #
    (lk.rules.on("email").str_endswith(pattern=".com"), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (~lk.rules.on("email").str_endswith(pattern=".com"), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    #
    (lk.rules.on("email").str_contains(pattern="@example"), [0, 1, 0, 0, 0, 0, 0, 0, 8, 0]),
    (~lk.rules.on("email").str_contains(pattern="@example"), [0, 1, 2, 3, 4, 5, 6, 7, 1, 9]),
    #
    (lk.rules.on("address").isna(), [0, 1, 2, 3, 4, 5, 6, 7, 4, 9]),
    (~lk.rules.on("address").isna(), [0, 0, 0, 0, 4, 0, 0, 0, 8, 0]),
]

# fmt: on


# Negation is strongly encouraged to be only for the Pipeline API!


@pytest.mark.parametrize("deduper, expected_canonical_id", PARAMS)
def test_matrix_negates(deduper, expected_canonical_id, dataframe, helpers):

    df, spark_session = dataframe

    df = (
        lk.Dedupe(df, spark_session=spark_session)
        .apply(lk.rules.pipeline().step(deduper))
        .canonicalize()
        .collect()
    )

    assert helpers.get_column_as_list(df, CANONICAL_ID) == expected_canonical_id
