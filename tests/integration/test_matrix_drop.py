"""TODO"""

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
    (strings_same_len, "email", {"drop_duplicates": False}, {"min_len": 3}, [0, 1, 2, 3, 2, 5, 2, 7, 3, 3, 10, 11, 3]),
    (strings_same_len, "email", {"drop_duplicates": True}, {"min_len": 3}, [0, 1, 2, 3, 5, 7, 10, 11]),
    # EXACT:
    # on single column
    (exact, SINGLE_COL, {"drop_duplicates": False}, {}, [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    (exact, SINGLE_COL, {"drop_duplicates": True}, {}, [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12]),
    # on compound column
    (exact, CATEGORICAL_COMPOUND_COL, {"drop_duplicates": False}, {}, [0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (exact, CATEGORICAL_COMPOUND_COL, {"drop_duplicates": True}, {}, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    #
    # FUZZY:
    (fuzzy, SINGLE_COL, {"drop_duplicates": False}, {"threshold": 0.65}, [0, 1, 2, 2, 4, 5, 6, 1, 0, 0, 10, 11, 12]),
    (fuzzy, SINGLE_COL, {"drop_duplicates": True}, {"threshold": 0.65}, [0, 1, 2, 4, 5, 6, 10, 11, 12]),
    #
    # COSINE:
    (cosine, NUMERICAL_COMPOUND_COL, {"drop_duplicates": False}, {"threshold": 0.99}, [0, 0, 0, 0, 4, 0, 0, 4, 8, 0, 0, 0, 0]),
    (cosine, NUMERICAL_COMPOUND_COL, {"drop_duplicates": True}, {"threshold": 0.99}, [0, 4, 8]),
    #
    # JACCARD:
    (jaccard, CATEGORICAL_COMPOUND_COL, {"drop_duplicates": False}, {"threshold": 0.35}, [0, 0, 2, 3, 4, 4, 0, 3, 8, 9, 0, 0, 4]),
    (jaccard, CATEGORICAL_COMPOUND_COL, {"drop_duplicates": True}, {"threshold": 0.35}, [0, 2, 3, 4, 8, 9]),
    #
    # LSH:
    (lsh, SINGLE_COL, {"drop_duplicates": False}, {"ngram": 1, "threshold": 0.65, "num_perm": 128}, [0, 1, 2, 2, 4, 5, 6, 1, 0, 0, 10, 11, 12]),
    (lsh, SINGLE_COL, {"drop_duplicates": True}, {"ngram": 1, "threshold": 0.65, "num_perm": 128}, [0, 1, 2, 4, 5, 6, 10, 11, 12]),
    #
    # STRING STARTS WITH:
    (str_startswith, SINGLE_COL, {"drop_duplicates": False}, {"pattern": "calle", "case": False}, [0, 1, 2, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (str_startswith, SINGLE_COL, {"drop_duplicates": True}, {"pattern": "calle", "case": False}, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    #
    # STRING ENDS WITH:
    (str_endswith, SINGLE_COL, {"drop_duplicates": False}, {"pattern": "kingdom", "case": False}, [0, 1, 2, 3, 1, 1, 6, 7, 8, 9, 10, 11, 12]),
    (str_endswith, SINGLE_COL, {"drop_duplicates": True}, {"pattern": "kingdom", "case": False}, [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12]),
    #
    # STRING CONTAINS:
    (str_contains, SINGLE_COL, {"drop_duplicates": False}, {"pattern": r"05\d{3}", "case": False, "regex": True}, [0, 1, 2, 2, 4, 5, 2, 7, 8, 9, 10, 11, 12]),
    (str_contains, SINGLE_COL, {"drop_duplicates": True}, {"pattern": r"05\d{3}", "case": False, "regex": True}, [0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12]),
    #
    # TF IDF:
    # progressive deduping: vary threshold
    (tfidf, SINGLE_COL, {"drop_duplicates": False}, {"ngram": 1, "threshold": 0.80, "topn": 2}, [0, 1, 2, 2, 4, 1, 2, 1, 0, 0, 4, 11, 12]),
    (tfidf, SINGLE_COL, {"drop_duplicates": True}, {"ngram": 1, "threshold": 0.80, "topn": 2}, [0, 1, 2, 4, 11, 12]),
]

# fmt: on

IDS = [
    "custom-no-drop",
    "custom-with-drop",
    "exact-single-no-drop",
    "exact-single-with-drop",
    "exact-compound-no-drop",
    "exact-compound-with-drop",
    "fuzzy-no-drop",
    "fuzzy-with-drop",
    "cosine-no-drop",
    "cosine-with-drop",
    "jaccard-no-drop",
    "jaccard-with-drop",
    "lsh-no-drop",
    "lsh-with-drop",
    "str_startswith-no-drop",
    "str_startswith-with-drop",
    "str_endswith-no-drop",
    "str_endswith-with-drop",
    "str_contains-no-drop",
    "str_contains-with-drop",
    "tfidf-no-drop",
    "tfidf-with-drop",
]

@pytest.mark.parametrize("strategy, columns, drop_kwarg, strat_kwarg, expected_canonical_id", PARAMS, ids=IDS)
def test_matrix_strats(strategy, columns, drop_kwarg, strat_kwarg, expected_canonical_id, dataframe, helpers):

    df, spark_session = dataframe

    dg = Duped(df, spark_session=spark_session)

    # single strategy item addition
    dg.apply(strategy(**strat_kwarg))
    dg.canonicalize(columns, **drop_kwarg)

    assert helpers.get_column_as_list(dg.df, CANONICAL_ID) == expected_canonical_id

    dg = Duped(df, spark_session=spark_session)

    # dictionary strategy addition
    dg.apply({columns: [strategy(**strat_kwarg)]})
    dg.canonicalize(**drop_kwarg)

    assert helpers.get_column_as_list(dg.df, CANONICAL_ID) == expected_canonical_id
