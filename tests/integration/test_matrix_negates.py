"""Narrow integration tests for predicate dedupers under negation"""

from __future__ import annotations

import pytest

import liken as lk
from liken.constants import CANONICAL_ID


# fmt: off

PARAMS = [
    #
    (lk.col("email").str_len(min_len=16, max_len=22), [0, 1, 2, 0, 4, 5, 0, 0, 8, 9]),
    (~lk.col("email").str_len(min_len=16, max_len=22), [0, 1, 1, 3, 1, 1, 6, 7, 1, 1]),
    #
    (lk.col("email").str_startswith(pattern="a"), [0, 1, 1, 3, 4, 5, 6, 7, 8, 9]),
    (~lk.col("email").str_startswith(pattern="a"), [0, 1, 2, 0, 0, 0, 0, 0, 0, 0]),
    #
    (lk.col("email").str_endswith(pattern=".com"), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (~lk.col("email").str_endswith(pattern=".com"), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    #
    (lk.col("email").str_contains(pattern="@example"), [0, 1, 0, 0, 0, 0, 0, 0, 8, 0]),
    (~lk.col("email").str_contains(pattern="@example"), [0, 1, 2, 3, 4, 5, 6, 7, 1, 9]),
    #
    (lk.col("address").isna(), [0, 1, 2, 3, 4, 5, 6, 7, 4, 9]),
    (~lk.col("address").isna(), [0, 0, 0, 0, 4, 0, 0, 0, 8, 0]),
    #
    (lk.col("address").isin(values=["123ab, OL5 9PL, UK"]), [0, 1, 2, 3, 4, 5, 6, 0, 8, 9]),
    (~lk.col("address").isin(values=["zzzzz"]), [0, 0, 0, 0, 4, 0, 0, 0, 8, 0]),
    #
    (lk.col("property_area_sq_ft").num_range(min=500, max=620), [0, 1, 2, 3, 4, 5, 0, 0, 8, 9]),
    (~lk.col("property_area_sq_ft").num_range(min=500, max=620), [0, 1, 1, 1, 1, 1, 6, 7, 1, 1]),
    #
    # ~date_range: no fixture value fails to parse under the format and
    # nulls match nothing on either polarity, so negation is a no-op here
    (lk.col("date_joined").date_range(format="%Y-%m-%d"), [0, 0, 2, 3, 4, 5, 6, 6, 4, 9]),
    (~lk.col("date_joined").date_range(format="%Y-%m-%d"), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
]

# fmt: on


# Negation is strongly encouraged to be only for the Pipeline API!


@pytest.mark.parametrize("deduper, expected_canonical_id", PARAMS)
def test_matrix_negates(deduper, expected_canonical_id, dataframe, helpers):

    df = (
        lk.dedupe(
            dataframe,
        )
        .apply(lk.pipeline().step(deduper))
        .canonicalize()
        .collect()
    )

    assert helpers.get_column_as_list(df, CANONICAL_ID) == expected_canonical_id
