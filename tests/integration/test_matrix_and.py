"""Narrow integration tests for specific behaviour of individual stratgies"""

from __future__ import annotations

import typing

import pytest

from dupegrouper import Dedupe, fuzzy
from dupegrouper._constants import CANONICAL_ID
from dupegrouper.custom import register
from dupegrouper.rules import Rules, on


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
    # single column
    ((on("email", fuzzy(0.95)),), [0, 0, 2, 3, 0, 0, 0, 3, 3, 9, 10, 11, 12]),
    ((on("email", fuzzy(0.95)) & on("email", str_same_len()),), [0, 1, 2, 3, 4, 5, 4, 7, 3, 9, 10, 11, 12]),
    # single column
    ((on("address", fuzzy(0.70)),), [0, 1, 2, 2, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    ((on("address", fuzzy(0.70)) & on("address", str_same_len()),), [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
]

# fmt: on


@pytest.mark.parametrize("strat, expected_canonical_id", PARAMS)
def test_matrix_and(strat, expected_canonical_id, dataframe, helpers):

    df, spark_session = dataframe

    dg = Dedupe(df, spark_session=spark_session)
    dg.apply(Rules(strat))
    dg.canonicalize()

    assert helpers.get_column_as_list(dg.df, CANONICAL_ID) == expected_canonical_id
