"""Narrow integration tests for specific behaviour of individual stratgies"""

from __future__ import annotations

import pytest

from dupegrouper import Duped
from dupegrouper.base import wrap
from dupegrouper.definitions import CANONICAL_ID
from dupegrouper.strategies import (
    Cosine,
    Exact,
    Fuzzy,
    Jaccard,
    Lsh,
    StrStartsWith,
    StrEndsWith,
    StrContains,
    TfIdf,
)


SINGLE_COL = "address"
CATEGORICAL_COMPOUND_COL = (
    "account",
    "birth_country",
    "martial_status",
    "number_children",
    "property_type",
)
NUMERICAL_COMPOUND_COL = (
    "property_height",
    "property_area_sq_ft",
    "property_sea_level_elevation_m",
    "property_num_rooms",
)

# fmt: off

PARAMS = [
    # EXACT:
    (Exact, "first", SINGLE_COL, {}, [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 11, 12, 13]),
    (Exact, "last", SINGLE_COL, {}, [10, 2, 3, 4, 5, 6, 7, 8, 10, 10, 11, 12, 13]),
    #
    # FUZZY:
    # no deduping: threshold applied at > not >=.
    (Fuzzy, "first", SINGLE_COL, {"threshold": 1}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    (Fuzzy, "last", SINGLE_COL, {"threshold": 1}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # progressive deduping: "first"
    (Fuzzy, "first", SINGLE_COL, {"threshold": 0.95}, [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 11, 12, 13]),
    (Fuzzy, "first", SINGLE_COL, {"threshold": 0.85}, [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 11, 12, 13]),
    (Fuzzy, "first", SINGLE_COL, {"threshold": 0.75}, [1, 2, 3, 3, 5, 6, 7, 8, 1, 1, 11, 12, 13]),
    (Fuzzy, "first", SINGLE_COL, {"threshold": 0.65}, [1, 2, 3, 3, 5, 6, 7, 2, 1, 1, 11, 12, 13]),
    (Fuzzy, "first", SINGLE_COL, {"threshold": 0.55}, [1, 2, 3, 3, 2, 6, 3, 2, 1, 1, 2, 12, 13]),
    (Fuzzy, "first", SINGLE_COL, {"threshold": 0.45}, [1, 2, 3, 3, 2, 2, 3, 2, 1, 1, 2, 1, 13]),
    (Fuzzy, "first", SINGLE_COL, {"threshold": 0.35}, [1, 2, 3, 3, 2, 1, 3, 2, 1, 1, 2, 2, 13]),
    (Fuzzy, "first", SINGLE_COL, {"threshold": 0.25}, [1, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1]),
    (Fuzzy, "first", SINGLE_COL, {"threshold": 0.15}, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    # progressive deduping: "last"
    # (Fuzzy, "last", SINGLE_COL, {"threshold": 0.95}, [10, 2, 3, 4, 5, 6, 7, 8, 10, 10, 11, 12, 13]),
    # (Fuzzy, "last", SINGLE_COL, {"threshold": 0.85}, [10, 2, 3, 4, 5, 6, 7, 8, 10, 10, 11, 12, 13]),
    # (Fuzzy, "last", SINGLE_COL, {"threshold": 0.75}, [10, 2, 4, 4, 5, 6, 7, 8, 10, 10, 11, 12, 13]),
    # (Fuzzy, "last", SINGLE_COL, {"threshold": 0.65}, [10, 8, 4, 4, 5, 6, 7, 8, 10, 10, 11, 12, 13]),
    # (Fuzzy, "last", SINGLE_COL, {"threshold": 0.55}, [10, 8, 7, 7, 11, 6, 7, 8, 10, 10, 11, 12, 13]),
    # (Fuzzy, "last", SINGLE_COL, {"threshold": 0.45}, [12, 11, 7, 7, 11, 6, 7, 8, 12, 12, 11, 12, 13]),
    # (Fuzzy, "last", SINGLE_COL, {"threshold": 0.35}, [12, 12, 7, 7, 12, 12, 7, 12, 12, 12, 12, 12, 13]),
    # (Fuzzy, "last", SINGLE_COL, {"threshold": 0.25}, [12, 11, 13, 13, 13, 13, 13, 13, 12, 12, 13, 12, 13]),
    # (Fuzzy, "last", SINGLE_COL, {"threshold": 0.15}, [13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 12, 13]),
    #
    # COSINE:
    # no deduping: threshold applied at > not >=.
    # (Cosine, "first", NUMERICAL_COMPOUND_COL, {"threshold": 1}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # (Cosine, "last", NUMERICAL_COMPOUND_COL, {"threshold": 1}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # # progressive deduping: "first"
    # (Cosine, "first", NUMERICAL_COMPOUND_COL, {"threshold": 0.999}, [1, 1, 1, 4, 5, 1, 7, 5, 9, 1, 1, 1, 1]),
    # (Cosine, "first", NUMERICAL_COMPOUND_COL, {"threshold": 0.99}, [1, 1, 1, 1, 5, 1, 1, 5, 9, 1, 1, 1, 1]),
    # (Cosine, "first", NUMERICAL_COMPOUND_COL, {"threshold": 0.98}, [1, 1, 1, 1, 5, 1, 1, 5, 5, 1, 1, 1, 1]),
    # (Cosine, "first", NUMERICAL_COMPOUND_COL, {"threshold": 0.95}, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    # progressive deduping: "last"
    # (Cosine, "last", NUMERICAL_COMPOUND_COL, {"threshold": 0.999}, [13, 13, 13, 4, 8, 13, 7, 8, 9, 13, 13, 13, 13]),
    # (Cosine, "last", NUMERICAL_COMPOUND_COL, {"threshold": 0.99}, [13, 13, 13, 13, 8, 13, 13, 8, 9, 13, 13, 13, 13]),
    # (Cosine, "last", NUMERICAL_COMPOUND_COL, {"threshold": 0.98}, [13, 13, 13, 13, 9, 13, 13, 9, 9, 13, 13, 13, 13]),
    # (Cosine, "last", NUMERICAL_COMPOUND_COL, {"threshold": 0.95}, [13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13]),
    #
    # # JACCARD:
    # # no deduping: threshold applied at > not >=.
    # (Jaccard, "first", CATEGORICAL_COMPOUND_COL, {"threshold": 1}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # # progressive deduping
    # (Jaccard, "first", CATEGORICAL_COMPOUND_COL, {"threshold": 0.65}, [1, 1, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 13]),
    # (Jaccard, "first", CATEGORICAL_COMPOUND_COL, {"threshold": 0.35}, [1, 1, 3, 4, 5, 5, 1, 4, 9, 10, 1, 1, 5]),
    # (Jaccard, "first", CATEGORICAL_COMPOUND_COL, {"threshold": 0.15}, [1, 1, 3, 1, 1, 1, 1, 1, 1, 10, 1, 1, 1]),
    # (Jaccard, "first", CATEGORICAL_COMPOUND_COL, {"threshold": 0.05}, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    # #
    # # LSH:
    # # progressive deduping: fix ngram; vary threshold
    # (Lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.95, "num_perm": 128}, [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 11, 12, 13]),
    # (Lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.85, "num_perm": 128}, [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 11, 12, 13]),
    # (Lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.75, "num_perm": 128}, [1, 2, 3, 4, 5, 6, 7, 2, 1, 1, 11, 12, 13]),
    # (Lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.65, "num_perm": 128}, [1, 2, 3, 3, 5, 6, 7, 2, 1, 1, 11, 12, 13]),
    # (Lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 128}, [1, 2, 3, 3, 5, 6, 3, 2, 1, 1, 2, 12, 13]),
    # (Lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.45, "num_perm": 128},  [1, 2, 3, 3, 2, 6, 3, 2, 1, 1, 2, 1, 2]),
    # (Lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.35, "num_perm": 128},  [1, 2, 2, 2, 2, 1, 2, 2, 1, 1, 1, 1, 2]),
    # (Lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.25, "num_perm": 128},  [1, 2, 2, 2, 2, 1, 2, 2, 1, 1, 2, 2, 2]),
    # (Lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.15, "num_perm": 128},  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    # # progressive deduping: fix threshold; vary ngram
    # (Lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.45, "num_perm": 128},  [1, 2, 3, 3, 2, 6, 3, 2, 1, 1, 2, 1, 2]),
    # (Lsh, "first", SINGLE_COL, {"ngram": 2, "threshold": 0.45, "num_perm": 128},  [1, 2, 3, 3, 5, 6, 7, 8, 1, 1, 11, 12, 13]),
    # (Lsh, "first", SINGLE_COL, {"ngram": 3, "threshold": 0.45, "num_perm": 128},  [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 11, 12, 13]),
    # # progressive deduping: fix parameters; vary permutations
    # (Lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 32}, [1, 2, 3, 3, 2, 2, 7, 2, 1, 1, 2, 12, 13]),
    # (Lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 64}, [1, 2, 3, 3, 2, 6, 7, 2, 1, 1, 2, 1, 2]),
    # (Lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 128}, [1, 2, 3, 3, 5, 6, 3, 2, 1, 1, 2, 12, 13]),
    # (Lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 256}, [1, 2, 3, 3, 5, 6, 3, 2, 1, 1, 2, 12, 13]),    
    # #
    # # STRING STARTS WITH:
    # # i.e. no deduping because no string starts with the pattern
    # (StrStartsWith, "first", SINGLE_COL, {"pattern": "zzzzz", "case": True}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # (StrStartsWith, "first", SINGLE_COL, {"pattern": "zzzzz", "case": False}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # # String does canonicalize if case correct; but doesn't otherwise
    # (StrStartsWith, "first", SINGLE_COL, {"pattern": "calle", "case": True}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # (StrStartsWith, "first", SINGLE_COL, {"pattern": "calle", "case": False}, [1, 2, 3, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # #
    # # STRING ENDS WITH:
    # # i.e. no deduping because no string starts with the pattern
    # (StrEndsWith, "first", SINGLE_COL, {"pattern": "zzzzz", "case": True}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # (StrEndsWith, "first", SINGLE_COL, {"pattern": "zzzzz", "case": False}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # # String does canonicalize if case correct; but doesn't otherwise
    # (StrEndsWith, "first", SINGLE_COL, {"pattern": "kingdom", "case": True}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # (StrEndsWith, "first", SINGLE_COL, {"pattern": "kingdom", "case": False}, [1, 2, 3, 4, 2, 2, 7, 8, 9, 10, 11, 12, 13]),
    # #
    # # STRING CONTAINS:
    # # i.e. no deduping because no string starts with the pattern
    # (StrContains, "first", SINGLE_COL, {"pattern": "zzzzz", "case": True, "regex": True}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # (StrContains, "first", SINGLE_COL, {"pattern": "zzzzz", "case": False, "regex": True}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # (StrContains, "first", SINGLE_COL, {"pattern": "zzzzz", "case": True, "regex": False}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # (StrContains, "first", SINGLE_COL, {"pattern": "zzzzz", "case": False, "regex": False}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # # String does canonicalize if case correct; but doesn't otherwise, no regex
    # (StrContains, "first", SINGLE_COL, {"pattern": "ol5 9pl", "case": True, "regex": False}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # (StrContains, "first", SINGLE_COL, {"pattern": "ol5 9pl", "case": False, "regex": False}, [1, 2, 3, 4, 5, 1, 7, 8, 1, 1, 11, 12, 13]),
    # # String does canonicalize if case correct; but doesn't otherwise, with regex
    # (StrContains, "first", SINGLE_COL, {"pattern": r"05\d{3}", "case": True, "regex": True}, [1, 2, 3, 3, 5, 6, 3, 8, 9, 10, 11, 12, 13]),
    # (StrContains, "first", SINGLE_COL, {"pattern": r"05\d{3}", "case": False, "regex": True}, [1, 2, 3, 3, 5, 6, 3, 8, 9, 10, 11, 12, 13]),
    # #
    # # TF IDF:
    # # no deduping: threshold applied at > not >=.
    # (TfIdf, "first", SINGLE_COL, {"ngram": (1, 1), "threshold": 1, "topn": 1}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # # progressive deduping: fix ngram; vary threshold
    # (TfIdf, "first", SINGLE_COL, {"ngram": (1, 1), "threshold": 0.95, "topn": 2}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # (TfIdf, "first", SINGLE_COL, {"ngram": (1, 1), "threshold": 0.80, "topn": 2}, [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 12, 13]),
    # (TfIdf, "first", SINGLE_COL, {"ngram": (1, 1), "threshold": 0.65, "topn": 2}, [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 12, 2]),
    # (TfIdf, "first", SINGLE_COL, {"ngram": (1, 1), "threshold": 0.50, "topn": 2}, [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 5, 2]),
    # (TfIdf, "first", SINGLE_COL, {"ngram": (1, 1), "threshold": 0.35, "topn": 2}, [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 5, 2]),
    # (TfIdf, "first", SINGLE_COL, {"ngram": (1, 1), "threshold": 0.15, "topn": 2}, [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 5, 2]),
    # # progressive deduping: fix threshold; vary ngram
    # (TfIdf, "first", SINGLE_COL, {"ngram": (1, 2), "threshold": 0.80, "topn": 2}, [1, 2, 3, 3, 5, 6, 7, 2, 1, 1, 11, 12, 13]),
    # (TfIdf, "first", SINGLE_COL, {"ngram": (1, 3), "threshold": 0.80, "topn": 2}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # (TfIdf, "first", SINGLE_COL, {"ngram": (2, 3), "threshold": 0.80, "topn": 2}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # (TfIdf, "first", SINGLE_COL, {"ngram": (2, 2), "threshold": 0.80, "topn": 2}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # (TfIdf, "first", SINGLE_COL, {"ngram": (3, 3), "threshold": 0.80, "topn": 2}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # # progressive deduping: fix threshold; vary ngram
    # (TfIdf, "first", SINGLE_COL, {"ngram": (1, 2), "threshold": 0.60, "topn": 2}, [1, 2, 3, 3, 2, 2, 3, 2, 1, 1, 11, 12, 13]),
    # (TfIdf, "first", SINGLE_COL, {"ngram": (1, 3), "threshold": 0.60, "topn": 2}, [1, 2, 3, 3, 5, 6, 7, 2, 1, 1, 11, 12, 13]),
    # (TfIdf, "first", SINGLE_COL, {"ngram": (2, 3), "threshold": 0.60, "topn": 2}, [1, 2, 3, 3, 5, 6, 7, 2, 1, 1, 11, 12, 13]),
    # (TfIdf, "first", SINGLE_COL, {"ngram": (2, 2), "threshold": 0.60, "topn": 2}, [1, 2, 3, 3, 5, 6, 7, 2, 1, 1, 11, 12, 13]),
    # (TfIdf, "first", SINGLE_COL, {"ngram": (3, 3), "threshold": 0.60, "topn": 2}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
]

# fmt: on


@pytest.mark.parametrize("strategy, rule, columns, input_params, expected_canonical_id", PARAMS)
def test_canonicalize_matrix(strategy, rule, columns, input_params, expected_canonical_id, dataframe, helpers):

    df, spark_session, id = dataframe

    dg = Duped(df=df, spark_session=spark_session, id=id, canonicalization_rule=rule)

    # single strategy item addition
    dg.apply(strategy(**input_params))
    dg.canonicalize(columns)
    df1 = dg.df

    assert helpers.get_column_as_list(df1, CANONICAL_ID) == expected_canonical_id

    # dictionary strategy addition
    dg.apply({columns: [strategy(**input_params)]})
    dg.canonicalize()
    df2 = dg.df

    assert helpers.get_column_as_list(df2, CANONICAL_ID) == expected_canonical_id