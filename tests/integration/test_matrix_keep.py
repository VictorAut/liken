"""Narrow integration tests for specific behaviour of individual stratgies"""

from __future__ import annotations

import typing

import pytest

from dupegrouper import (
    Duped,
    cosine,
    exact,
    fuzzy,
    jaccard,
    lsh,
    tfidf,
)
from dupegrouper._constants import CANONICAL_ID
from dupegrouper.custom import register
from dupegrouper.rules import Rules, on, str_contains, str_endswith, str_startswith

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
def strings_same_len(array: typing.Iterable, min_len: int = 3):
    n = len(array)
    for i in range(n):
        for j in range(i + 1, n):
            if (
                len(array[i]) >= min_len
                and len(array[j]) >= min_len
                #
                and len(array[i]) == len(array[j])
            ):
                yield i, j


# fmt: off

PARAMS = [
    # CUSTOM:
    (strings_same_len, "first", "email", {"min_len": 3}, [0, 1, 2, 3, 2, 5, 2, 7, 3, 3, 10, 11, 3]),
    (strings_same_len, "last", "email", {"min_len": 3}, [0, 1, 6, 12, 6, 5, 6, 7, 12, 12, 10, 11, 12]),
    # EXACT:
    (exact, "first", SINGLE_COL, {}, [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    (exact, "last", SINGLE_COL, {}, [9, 1, 2, 3, 4, 5, 6, 7, 9, 9, 10, 11, 12]),
    (exact, "first", CATEGORICAL_COMPOUND_COL, {}, [0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (exact, "last", CATEGORICAL_COMPOUND_COL, {}, [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    #
    # FUZZY:
    (fuzzy, "first", SINGLE_COL, {"threshold": 0.65}, [0, 1, 2, 2, 4, 5, 6, 1, 0, 0, 10, 11, 12]),
    (fuzzy, "last", SINGLE_COL, {"threshold": 0.65}, [9, 7, 3, 3, 4, 5, 6, 7, 9, 9, 10, 11, 12]),
    #
    # COSINE:
    (cosine, "first", NUMERICAL_COMPOUND_COL, {"threshold": 0.99}, [0, 0, 0, 0, 4, 0, 0, 4, 8, 0, 0, 0, 0]),
    (cosine, "last", NUMERICAL_COMPOUND_COL, {"threshold": 0.99}, [12, 12, 12, 12, 7, 12, 12, 7, 8, 12, 12, 12, 12]),
    #
    # JACCARD:
    (jaccard, "first", CATEGORICAL_COMPOUND_COL, {"threshold": 0.65}, [0, 0, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 12]),
    (jaccard, "last", CATEGORICAL_COMPOUND_COL, {"threshold": 0.65}, [6, 6, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    #
    # LSH:
    (lsh, "first", SINGLE_COL, {"ngram": 2, "threshold": 0.45, "num_perm": 128},  [0, 1, 2, 2, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    (lsh, "last", SINGLE_COL, {"ngram": 2, "threshold": 0.45, "num_perm": 128},  [9, 1, 3, 3, 4, 5, 6, 7, 9, 9, 10, 11, 12]),
    #
    # STRING STARTS WITH:
    (str_startswith, "first", SINGLE_COL, {"pattern": "calle", "case": False}, [0, 1, 2, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (str_startswith, "last", SINGLE_COL, {"pattern": "calle", "case": False}, [0, 1, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    #
    # STRING ENDS WITH:
    (str_endswith, "first", SINGLE_COL, {"pattern": "kingdom", "case": False}, [0, 1, 2, 3, 1, 1, 6, 7, 8, 9, 10, 11, 12]),
    (str_endswith, "last", SINGLE_COL, {"pattern": "kingdom", "case": False}, [0, 5, 2, 3, 5, 5, 6, 7, 8, 9, 10, 11, 12]),
    #
    # STRING CONTAINS:
    (str_contains, "first", SINGLE_COL, {"pattern": "ol5 9pl", "case": False, "regex": False}, [0, 1, 2, 3, 4, 0, 6, 7, 0, 0, 10, 11, 12]),
    (str_contains, "last", SINGLE_COL, {"pattern": "ol5 9pl", "case": False, "regex": False}, [9, 1, 2, 3, 4, 9, 6, 7, 9, 9, 10, 11, 12]),
    #
    # TF IDF:
    (tfidf, "first", SINGLE_COL, {"ngram": (1, 2), "threshold": 0.80, "topn": 2}, [0, 1, 2, 2, 4, 5, 6, 1, 0, 0, 10, 11, 12]),
    (tfidf, "last", SINGLE_COL, {"ngram": (1, 2), "threshold": 0.80, "topn": 2}, [9, 7, 3, 3, 4, 5, 6, 7, 9, 9, 10, 11, 12]),
]

# fmt: on


@pytest.mark.parametrize("strategy, keep, columns, input_params, expected_canonical_id", PARAMS)
def test_matrix_keep_sequence_api(strategy, keep, columns, input_params, expected_canonical_id, dataframe, helpers):

    df, spark_session = dataframe

    dg = Duped(df, spark_session=spark_session)
    dg.apply(strategy(**input_params))
    dg.canonicalize(columns, keep=keep)

    assert helpers.get_column_as_list(dg.df, CANONICAL_ID) == expected_canonical_id


@pytest.mark.parametrize("strategy, keep, columns, input_params, expected_canonical_id", PARAMS)
def test_matrix_keep_dict_api(strategy, keep, columns, input_params, expected_canonical_id, dataframe, helpers):

    df, spark_session = dataframe

    dg = Duped(df, spark_session=spark_session)
    dg.apply({columns: [strategy(**input_params)]})
    dg.canonicalize(keep=keep)

    assert helpers.get_column_as_list(dg.df, CANONICAL_ID) == expected_canonical_id


@pytest.mark.parametrize("strategy, keep, columns, input_params, expected_canonical_id", PARAMS)
def test_matrix_keep_rules_api(strategy, keep, columns, input_params, expected_canonical_id, dataframe, helpers):

    df, spark_session = dataframe

    dg = Duped(df, spark_session=spark_session)
    dg.apply(Rules(on(columns, strategy(**input_params))))
    dg.canonicalize(keep=keep)

    assert helpers.get_column_as_list(dg.df, CANONICAL_ID) == expected_canonical_id
