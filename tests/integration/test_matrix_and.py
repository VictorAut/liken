"""Narrow integration tests for specific behaviour of individual stratgies"""

from __future__ import annotations

import typing

import pytest

from liken import Dedupe
from liken import exact
from liken import fuzzy
from liken._constants import CANONICAL_ID
from liken.custom import register
from liken.rules import Rules
from liken.rules import isna
from liken.rules import on


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


@register
def str_same_len(array: typing.Iterable):
    n = len(array)
    for i in range(n):
        for j in range(i + 1, n):
            if len(array[i]) == len(array[j]):
                yield i, j


# fmt: off

PARAMS = [
    # # single column
    # ((on("email").fuzzy(0.95),), [0, 1, 2, 3, 4, 4, 3, 3, 8, 0]),
    # ((on("email").fuzzy(0.95) & on("email").str_same_len(),),  [0, 1, 2, 3, 4, 4, 6, 3, 8, 9]),
    # # single column
    # ((on("address").fuzzy(0.70),), [0, 1, 2, 2, 4, 5, 6, 0, 4, 9]),
    # ((on("address").fuzzy(0.70) & on("address").str_same_len(),), [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    # single column
    ((on("address").fuzzy(0.70),), [0, 1, 2, 2, 4, 5, 6, 0, 4, 9]),
    ((on("address").fuzzy(0.70) & ~on("address").isna(),), [0, 1, 2, 2, 4, 5, 6, 0, 8, 9]),
    # single column
    ((on("account").exact(),), [0, 0, 2, 3, 4, 0, 0, 2, 8, 8]),
    ((on("property_height").isna() & on("account").exact(),), [0, 0, 2, 3, 4, 5, 6, 7, 8, 9]),
    # two threshold dedupers
    ((on("birth_country").exact() & on("marital_status").exact(),), [0, 0, 2, 3, 4, 3, 6, 7, 6, 9]),
]

# fmt: on


@pytest.mark.parametrize("strat, expected_canonical_id", PARAMS)
def test_matrix_and(strat, expected_canonical_id, dataframe, helpers):

    df, spark_session = dataframe

    df = (
        Dedupe(df, spark_session=spark_session)
        .apply(Rules(strat))
        .canonicalize()
        .collect()
    )

    assert helpers.get_column_as_list(df, CANONICAL_ID) == expected_canonical_id
