"""Narrow integration tests for specific behaviour of individual dedupers
when used as and combinations in a pipeline step."""

from __future__ import annotations

import typing

import pytest

import liken as lk
from liken.constants import CANONICAL_ID

# CONSTANTS:


SINGLE_COL = "address"
CATEGORICAL_COMPOUND_COL = (
    "account",
    "birth_country",
    "marital_status",
    "number_children",
    "property_type",
)
NUMERICAL_COMPOUND_COL = (
    "property_height",
    "property_area_sq_ft",
    "property_sea_level_elevation_m",
    "property_num_rooms",
)


# REGISTER A CUSTOM CALLABLE:


@lk.custom.register
def str_same_len(array: typing.Iterable):
    n = len(array)
    for i in range(n):
        for j in range(i + 1, n):
            if len(array[i]) == len(array[j]):
                yield i, j


# fmt: off

PARAMS = [
    # single column
    ([lk.col("email").fuzzy(0.95)], [0, 1, 2, 3, 4, 4, 3, 3, 8, 0]),
    ([lk.col("email").fuzzy(0.95), lk.col("email").str_same_len()],  [0, 1, 2, 3, 4, 4, 6, 3, 8, 9]),
    # single column
    ([lk.col("address").fuzzy(0.70)], [0, 1, 2, 2, 4, 5, 6, 0, 4, 9]),
    ([lk.col("address").fuzzy(0.70), lk.col("address").str_same_len()], [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    # single column
    ([lk.col("address").fuzzy(0.70)], [0, 1, 2, 2, 4, 5, 6, 0, 4, 9]),
    ([lk.col("address").fuzzy(0.70), ~lk.col("address").isna()], [0, 1, 2, 2, 4, 5, 6, 0, 8, 9]),
    # single column
    ([lk.col("account").exact()], [0, 0, 2, 3, 4, 0, 0, 2, 8, 8]),
    ([lk.col("property_height").isna(), lk.col("account").exact()], [0, 0, 2, 3, 4, 5, 6, 7, 8, 9]),
    # two threshold dedupers
    ([lk.col("birth_country").exact(), lk.col("marital_status").exact()], [0, 0, 2, 3, 4, 3, 6, 7, 6, 9]),
]

# fmt: on


@pytest.mark.parametrize("step, expected_canonical_id", PARAMS)
def test_matrix_and(step, expected_canonical_id, dataframe, helpers, spark_session):

    df = (
        lk.dedupe(
            dataframe,
            spark_session=spark_session,
        )
        .apply(lk.pipeline().step(step))
        .canonicalize()
        .collect()
    )

    assert helpers.get_column_as_list(df, CANONICAL_ID) == expected_canonical_id
