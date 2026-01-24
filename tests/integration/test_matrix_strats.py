"""Narrow integration tests for specific behaviour of individual stratgies"""

from __future__ import annotations

import typing

import pytest

from dupegrouper import (
    Dedupe,
    cosine,
    exact,
    fuzzy,
    jaccard,
    lsh,
    tfidf,
)
from enlace._constants import CANONICAL_ID
from enlace.custom import register
from enlace.rules import (
    Rules,
    isna,
    on,
    str_contains,
    str_endswith,
    str_len,
    str_startswith,
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
    (strings_same_len, "email", {"min_len": 3}, [0, 1, 2, 3, 2, 2, 6, 3, 8, 9]),
    (strings_same_len, "email", {"min_len": 15}, [0, 1, 2, 3, 4, 5, 6, 3, 8, 9]),
    # EXACT:
    # on single column
    (exact, SINGLE_COL, {}, [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    # on compound column
    (exact, CATEGORICAL_COMPOUND_COL, {}, [0, 0, 2, 3, 4, 5, 6, 7, 8, 9]),
    #
    # FUZZY:
    (fuzzy, SINGLE_COL, {"threshold": 0.95}, [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    (fuzzy, SINGLE_COL, {"threshold": 0.85}, [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    (fuzzy, SINGLE_COL, {"threshold": 0.75}, [0, 1, 2, 2, 4, 5, 6, 0, 4, 9]),
    (fuzzy, SINGLE_COL, {"threshold": 0.65}, [0, 1, 2, 2, 4, 5, 1, 0, 4, 9]),
    (fuzzy, SINGLE_COL, {"threshold": 0.55}, [0, 1, 2, 2, 4, 2, 1, 0, 4, 9]),
    (fuzzy, SINGLE_COL, {"threshold": 0.45}, [0, 1, 2, 2, 4, 2, 1, 0, 4, 1]),
    (fuzzy, SINGLE_COL, {"threshold": 0.35}, [0, 0, 2, 2, 4, 2, 0, 0, 4, 0]),
    (fuzzy, SINGLE_COL, {"threshold": 0.25}, [0, 0, 0, 0, 4, 0, 0, 0, 4, 0]),
    #
    # COSINE:
    (cosine, NUMERICAL_COMPOUND_COL, {"threshold": 0.999}, [0, 0, 0, 3, 0, 0, 6, 7, 0, 0]),
    (cosine, NUMERICAL_COMPOUND_COL, {"threshold": 0.99}, [0, 0, 0, 0, 0, 0, 6, 7, 0, 0]),
    (cosine, NUMERICAL_COMPOUND_COL, {"threshold": 0.98}, [0, 0, 0, 0, 0, 0, 6, 6, 0, 0]),
    #
    # JACCARD:
    (jaccard, CATEGORICAL_COMPOUND_COL, {"threshold": 0.65}, [0, 0, 2, 3, 4, 0, 6, 7, 8, 9]),
    (jaccard, CATEGORICAL_COMPOUND_COL, {"threshold": 0.35}, [0, 0, 2, 3, 0, 0, 3, 7, 0, 9]),
    #
    # LSH:
    # progressive deduping: fix ngram; vary threshold
    (lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.95, "num_perm": 128}, [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    (lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.85, "num_perm": 128}, [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    (lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.75, "num_perm": 128}, [0, 1, 2, 3, 4, 5, 1, 0, 4, 9]),
    (lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.65, "num_perm": 128}, [0, 1, 2, 2, 4, 5, 1, 0, 4, 9]),
    (lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 128}, [0, 1, 2, 2, 4, 2, 1, 0, 4, 9]),
    (lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.45, "num_perm": 128}, [0, 1, 2, 2, 4, 2, 1, 0, 4, 9]),
    (lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.35, "num_perm": 128}, [0, 1, 1, 1, 4, 1, 1, 0, 4, 0]),
    # progressive deduping: fix threshold; vary ngram
    (lsh, SINGLE_COL, {"ngram": 2, "threshold": 0.45, "num_perm": 128}, [0, 1, 2, 2, 4, 5, 6, 0, 4, 9]),
    (lsh, SINGLE_COL, {"ngram": 3, "threshold": 0.45, "num_perm": 128}, [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    # progressive deduping: fix parameters; vary permutations
    (lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 32}, [0, 1, 2, 2, 4, 5, 1, 0, 4, 9]),
    (lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 64}, [0, 1, 2, 2, 4, 5, 1, 0, 4, 9]),
    (lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 128}, [0, 1, 2, 2, 4, 2, 1, 0, 4, 9]),
    #
    # STRING STARTS WITH:
    # i.e. no deduping because no string starts with the pattern
    (str_startswith, SINGLE_COL, {"pattern": "zzzzz", "case": True}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    (str_startswith, SINGLE_COL, {"pattern": "zzzzz", "case": False}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    # String does canonicalize if case correct; but doesn't otherwise
    (str_startswith, SINGLE_COL, {"pattern": "calle", "case": True}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    (str_startswith, SINGLE_COL, {"pattern": "calle", "case": False}, [0, 1, 2, 2, 4, 5, 6, 7, 8, 9]),
    #
    # STRING ENDS WITH:
    # i.e. no deduping because no string starts with the pattern
    (str_endswith, SINGLE_COL, {"pattern": "zzzzz", "case": True}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    (str_endswith, SINGLE_COL, {"pattern": "zzzzz", "case": False}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    # String does canonicalize if case correct; but doesn't otherwise
    (str_endswith, SINGLE_COL, {"pattern": "kingdom", "case": True}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    (str_endswith, SINGLE_COL, {"pattern": "kingdom", "case": False}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 1]),
    #
    # STRING LEN:
    # i.e. no deduping because no such thing as max_len and min_len being inversered
    (str_len, "email", {"min_len": 10, "max_len": 9}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    # no deduping because bounds are out of range
    (str_len, "email", {"min_len": 101, "max_len": 201}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    # total deduping given no bounds
    (str_len, "email", {}, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    # resasonable bounds, given the column
    (str_len, "email", {"min_len": 15, "max_len": 22}, [0, 1, 2, 0, 4, 5, 0, 0, 8, 9]),
    #
    # STRING CONTAINS:
    # i.e. no deduping because no string starts with the pattern
    (str_contains, SINGLE_COL, {"pattern": "zzzzz", "case": True, "regex": True}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    (str_contains, SINGLE_COL, {"pattern": "zzzzz", "case": False, "regex": True}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    (str_contains, SINGLE_COL, {"pattern": "zzzzz", "case": True, "regex": False}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    (str_contains, SINGLE_COL, {"pattern": "zzzzz", "case": False, "regex": False}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    # String does canonicalize if case correct; but doesn't otherwise, no regex
    (str_contains, SINGLE_COL, {"pattern": "ol5 9pl", "case": True, "regex": False}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    (str_contains, SINGLE_COL, {"pattern": "ol5 9pl", "case": False, "regex": False}, [0, 1, 2, 3, 4, 5, 6, 0, 8, 0]),
    # String does canonicalize if case correct; but doesn't otherwise, with regex
    (str_contains, SINGLE_COL, {"pattern": r"05\d{3}", "case": True, "regex": True}, [0, 1, 2, 2, 4, 2, 6, 7, 8, 9]),
    (str_contains, SINGLE_COL, {"pattern": r"05\d{3}", "case": False, "regex": True}, [0, 1, 2, 2, 4, 2, 6, 7, 8, 9]),
    #
    # TF IDF:
    # progressive deduping: vary threshold
    (tfidf, SINGLE_COL, {"ngram": 1, "threshold": 0.95, "topn": 2}, [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    (tfidf, SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 2}, [0, 1, 2, 2, 4, 5, 1, 0, 4, 1]),
    (tfidf, SINGLE_COL, {"ngram": 1, "threshold": 0.65, "topn": 2}, [0, 1, 2, 2, 4, 2, 1, 0, 4, 1]),
    (tfidf, SINGLE_COL, {"ngram": 1, "threshold": 0.50, "topn": 2}, [0, 1, 2, 2, 4, 2, 1, 0, 4, 1]),
    # progressive deduping: vary ngram
    (tfidf, SINGLE_COL, {"ngram": (1, 2), "threshold": 0.80, "topn": 2}, [0, 1, 2, 2, 4, 5, 1, 0, 4, 9]),
    (tfidf, SINGLE_COL, {"ngram": (1, 3), "threshold": 0.80, "topn": 2}, [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    (tfidf, SINGLE_COL, {"ngram": (2, 3), "threshold": 0.80, "topn": 2}, [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    # progressive deduping: vary topn
    (tfidf, SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 1}, [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    (tfidf, SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 2}, [0, 1, 2, 2, 4, 5, 1, 0, 4, 1]),
    (tfidf, SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 3}, [0, 1, 2, 2, 4, 5, 1, 0, 4, 1]),
    #
    # ISNA:
    (isna, SINGLE_COL, {}, [0, 1, 2, 3, 4, 5, 6, 7, 4, 9]),
]

# fmt: on


@pytest.mark.parametrize("strategy, columns, strat_kwarg, expected_canonical_id", PARAMS)
def test_matrix_strats_sequence_api(strategy, columns, strat_kwarg, expected_canonical_id, dataframe, helpers):

    df, spark_session = dataframe

    dp = Dedupe(df, spark_session=spark_session)
    dp.apply(strategy(**strat_kwarg))
    df = dp.canonicalize(columns)

    assert helpers.get_column_as_list(df, CANONICAL_ID) == expected_canonical_id


@pytest.mark.parametrize("strategy, columns, strat_kwarg, expected_canonical_id", PARAMS)
def test_matrix_strats_dict_api(strategy, columns, strat_kwarg, expected_canonical_id, dataframe, helpers):

    df, spark_session = dataframe

    dp = Dedupe(df, spark_session=spark_session)
    dp.apply({columns: [strategy(**strat_kwarg)]})
    df = dp.canonicalize()

    assert helpers.get_column_as_list(df, CANONICAL_ID) == expected_canonical_id


@pytest.mark.parametrize("strategy, columns, strat_kwarg, expected_canonical_id", PARAMS)
def test_matrix_strats_rules_api(strategy, columns, strat_kwarg, expected_canonical_id, dataframe, helpers):

    df, spark_session = dataframe

    dp = Dedupe(df, spark_session=spark_session)
    dp.apply(Rules(on(columns, strategy(**strat_kwarg))))
    df = dp.canonicalize()

    assert helpers.get_column_as_list(df, CANONICAL_ID) == expected_canonical_id
