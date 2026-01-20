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
            if (
                len(array[i]) == len(array[j])
            ):
                yield i, j


# fmt: off

PARAMS = [
    # single column
    ((On("email", fuzzy(0.95)),), [0, 0, 2, 3, 0, 0, 0, 3, 3, 9, 10, 11, 12]),
    ((On("email", fuzzy(0.95)) & On("email", str_same_len()),), [0, 1, 2, 3, 4, 5, 4, 7, 3, 9, 10, 11, 12]),
    # single column
    # ((On("address", fuzzy(0.95)),), [0, 0, 2, 3, 0, 0, 0, 3, 3, 9, 10, 11, 12]),
    # ((On("address", fuzzy(0.95)) & On("address", str_same_len()),), [0, 1, 2, 3, 4, 5, 4, 7, 3, 9, 10, 11, 12]),
]

# fmt: on


@pytest.mark.parametrize("strat, expected_canonical_id", PARAMS)
def test_matrix_and(strat, expected_canonical_id, dataframe, helpers):

    df, spark_session = dataframe

    dg = Duped(df, spark_session=spark_session)
    dg.apply(strat)
    dg.canonicalize()

    assert helpers.get_column_as_list(dg.df, CANONICAL_ID) == expected_canonical_id