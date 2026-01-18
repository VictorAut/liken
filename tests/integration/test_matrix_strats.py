"""Narrow integration tests for specific behaviour of individual stratgies"""

from __future__ import annotations

import typing

import pytest

from dupegrouper import Duped
from dupegrouper.constants import CANONICAL_ID
from dupegrouper.custom import register
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
    (strings_same_len, "email", {"min_len": 3}, [0, 1, 2, 3, 2, 5, 2, 7, 3, 3, 10, 11, 3]),
    (strings_same_len, "email", {"min_len": 15}, [0, 1, 2, 3, 4, 5, 6, 7, 3, 3, 10, 11, 3]),
    # EXACT:
    # on single column
    (exact, SINGLE_COL, {}, [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    # on compound column
    (exact, CATEGORICAL_COMPOUND_COL, {}, [0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    #
    # FUZZY:
    (fuzzy, SINGLE_COL, {"threshold": 0.95}, [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    (fuzzy, SINGLE_COL, {"threshold": 0.85}, [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    (fuzzy, SINGLE_COL, {"threshold": 0.75}, [0, 1, 2, 2, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    (fuzzy, SINGLE_COL, {"threshold": 0.65}, [0, 1, 2, 2, 4, 5, 6, 1, 0, 0, 10, 11, 12]),
    (fuzzy, SINGLE_COL, {"threshold": 0.55}, [0, 1, 2, 2, 1, 5, 2, 1, 0, 0, 1, 11, 12]),
    (fuzzy, SINGLE_COL, {"threshold": 0.45}, [0, 1, 2, 2, 1, 1, 2, 1, 0, 0, 1, 0, 12]),
    (fuzzy, SINGLE_COL, {"threshold": 0.35}, [0, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 12]),
    (fuzzy, SINGLE_COL, {"threshold": 0.25}, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    #
    # COSINE:
    (cosine, NUMERICAL_COMPOUND_COL, {"threshold": 0.999}, [0, 0, 0, 3, 4, 0, 0, 4, 8, 0, 0, 0, 0]),
    (cosine, NUMERICAL_COMPOUND_COL, {"threshold": 0.99}, [0, 0, 0, 0, 4, 0, 0, 4, 8, 0, 0, 0, 0]),
    (cosine, NUMERICAL_COMPOUND_COL, {"threshold": 0.98}, [0, 0, 0, 0, 4, 0, 0, 4, 4, 0, 0, 0, 0]),
    (cosine, NUMERICAL_COMPOUND_COL, {"threshold": 0.95}, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    #
    # JACCARD:
    (jaccard, CATEGORICAL_COMPOUND_COL, {"threshold": 0.65}, [0, 0, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 12]),
    (jaccard, CATEGORICAL_COMPOUND_COL, {"threshold": 0.35}, [0, 0, 2, 3, 4, 4, 0, 3, 8, 9, 0, 0, 4]),
    (jaccard, CATEGORICAL_COMPOUND_COL, {"threshold": 0.15}, [0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0]),
    (jaccard, CATEGORICAL_COMPOUND_COL, {"threshold": 0.05}, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    #
    # LSH:
    # progressive deduping: fix ngram; vary threshold
    (lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.95, "num_perm": 128}, [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    (lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.85, "num_perm": 128}, [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    (lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.75, "num_perm": 128}, [0, 1, 2, 3, 4, 5, 6, 1, 0, 0, 10, 11, 12]),
    (lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.65, "num_perm": 128}, [0, 1, 2, 2, 4, 5, 6, 1, 0, 0, 10, 11, 12]),
    (lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 128}, [0, 1, 2, 2, 4, 5, 2, 1, 0, 0, 1, 11, 12]),
    (lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.45, "num_perm": 128},  [0, 1, 2, 2, 1, 5, 2, 1, 0, 0, 1, 0, 1]),
    (lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.35, "num_perm": 128},  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    # progressive deduping: fix threshold; vary ngram
    (lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.45, "num_perm": 128},  [0, 1, 2, 2, 1, 5, 2, 1, 0, 0, 1, 0, 1]),
    (lsh, SINGLE_COL, {"ngram": 2, "threshold": 0.45, "num_perm": 128},  [0, 1, 2, 2, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    (lsh, SINGLE_COL, {"ngram": 3, "threshold": 0.45, "num_perm": 128},  [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    # progressive deduping: fix parameters; vary permutations
    (lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 32}, [0, 1, 2, 2, 1, 1, 6, 1, 0, 0, 1, 11, 12]),
    (lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 64}, [0, 1, 2, 2, 1, 5, 6, 1, 0, 0, 1, 0, 1]),
    (lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 128}, [0, 1, 2, 2, 4, 5, 2, 1, 0, 0, 1, 11, 12]),
    (lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 256}, [0, 1, 2, 2, 4, 5, 2, 1, 0, 0, 1, 11, 12]),  
    #
    # STRING STARTS WITH:
    # i.e. no deduping because no string starts with the pattern
    (str_startswith, SINGLE_COL, {"pattern": "zzzzz", "case": True}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (str_startswith, SINGLE_COL, {"pattern": "zzzzz", "case": False}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    # String does canonicalize if case correct; but doesn't otherwise
    (str_startswith, SINGLE_COL, {"pattern": "calle", "case": True}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (str_startswith, SINGLE_COL, {"pattern": "calle", "case": False}, [0, 1, 2, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    #
    # STRING ENDS WITH:
    # i.e. no deduping because no string starts with the pattern
    (str_endswith, SINGLE_COL, {"pattern": "zzzzz", "case": True}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (str_endswith, SINGLE_COL, {"pattern": "zzzzz", "case": False}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    # String does canonicalize if case correct; but doesn't otherwise
    (str_endswith, SINGLE_COL, {"pattern": "kingdom", "case": True}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (str_endswith, SINGLE_COL, {"pattern": "kingdom", "case": False}, [0, 1, 2, 3, 1, 1, 6, 7, 8, 9, 10, 11, 12]),
    #
    # STRING CONTAINS:
    # i.e. no deduping because no string starts with the pattern
    (str_contains, SINGLE_COL, {"pattern": "zzzzz", "case": True, "regex": True}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (str_contains, SINGLE_COL, {"pattern": "zzzzz", "case": False, "regex": True}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (str_contains, SINGLE_COL, {"pattern": "zzzzz", "case": True, "regex": False}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (str_contains, SINGLE_COL, {"pattern": "zzzzz", "case": False, "regex": False}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    # String does canonicalize if case correct; but doesn't otherwise, no regex
    (str_contains, SINGLE_COL, {"pattern": "ol5 9pl", "case": True, "regex": False}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (str_contains, SINGLE_COL, {"pattern": "ol5 9pl", "case": False, "regex": False}, [0, 1, 2, 3, 4, 0, 6, 7, 0, 0, 10, 11, 12]),
    # String does canonicalize if case correct; but doesn't otherwise, with regex
    (str_contains, SINGLE_COL, {"pattern": r"05\d{3}", "case": True, "regex": True}, [0, 1, 2, 2, 4, 5, 2, 7, 8, 9, 10, 11, 12]),
    (str_contains, SINGLE_COL, {"pattern": r"05\d{3}", "case": False, "regex": True}, [0, 1, 2, 2, 4, 5, 2, 7, 8, 9, 10, 11, 12]),
    #
    # TF IDF:
    # progressive deduping: vary threshold
    (tfidf, SINGLE_COL, {"ngram": 1, "threshold": 0.95, "topn": 2}, [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    (tfidf, SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 2}, [0, 1, 2, 2, 4, 1, 2, 1, 0, 0, 4, 11, 12]),
    (tfidf, SINGLE_COL, {"ngram": 1, "threshold": 0.65, "topn": 2}, [0, 1, 2, 2, 4, 1, 2, 1, 0, 0, 4, 11, 1]),
    (tfidf, SINGLE_COL, {"ngram": 1, "threshold": 0.50, "topn": 2}, [0, 1, 2, 2, 4, 1, 2, 1, 0, 0, 4, 4, 1]),
    # progressive deduping: vary ngram
    (tfidf, SINGLE_COL, {"ngram": (1, 2), "threshold": 0.80, "topn": 2}, [0, 1, 2, 2, 4, 5, 6, 1, 0, 0, 10, 11, 12]),
    (tfidf, SINGLE_COL, {"ngram": (1, 3), "threshold": 0.80, "topn": 2}, [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    (tfidf, SINGLE_COL, {"ngram": (2, 3), "threshold": 0.80, "topn": 2}, [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    # progressive deduping: vary topn
    (tfidf, SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 1}, [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    (tfidf, SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 2}, [0, 1, 2, 2, 4, 1, 2, 1, 0, 0, 4, 11, 12]),
    (tfidf, SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 3}, [0, 1, 2, 2, 1, 1, 2, 1, 0, 0, 1, 11, 12]),
]

# fmt: on


@pytest.mark.parametrize("strategy, columns, strat_kwarg, expected_canonical_id", PARAMS)
def test_matrix_strats(strategy, columns, strat_kwarg, expected_canonical_id, dataframe, helpers):

    df, spark_session = dataframe

    dg = Duped(df, spark_session=spark_session)

    # single strategy item addition
    dg.apply(strategy(**strat_kwarg))
    dg.canonicalize(columns)

    assert helpers.get_column_as_list(dg.df, CANONICAL_ID) == expected_canonical_id

    dg = Duped(df, spark_session=spark_session)

    # dictionary strategy addition
    dg.apply({columns: [strategy(**strat_kwarg)]})
    dg.canonicalize()

    assert helpers.get_column_as_list(dg.df, CANONICAL_ID) == expected_canonical_id
