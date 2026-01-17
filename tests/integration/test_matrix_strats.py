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
    (strings_same_len, "first", "email", {"min_len": 3}, [0, 1, 2, 3, 2, 5, 2, 7, 3, 3, 10, 11, 3]),
    (strings_same_len, "last", "email", {"min_len": 3}, [0, 1, 6, 12, 6, 5, 6, 7, 12, 12, 10, 11, 12]),
    (strings_same_len, "first", "email", {"min_len": 15}, [0, 1, 2, 3, 4, 5, 6, 7, 3, 3, 10, 11, 3]),
    (strings_same_len, "last", "email", {"min_len": 15}, [0, 1, 2, 12, 4, 5, 6, 7, 12, 12, 10, 11, 12]),
    # EXACT:
    # on single column
    (exact, "first", SINGLE_COL, {}, [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    (exact, "last", SINGLE_COL, {}, [9, 1, 2, 3, 4, 5, 6, 7, 9, 9, 10, 11, 12]),
    # on compound column
    (exact, "first", CATEGORICAL_COMPOUND_COL, {}, [0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (exact, "last", CATEGORICAL_COMPOUND_COL, {}, [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    #
    # FUZZY:
    # progressive deduping: "first"
    (fuzzy, "first", SINGLE_COL, {"threshold": 0.95}, [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    (fuzzy, "first", SINGLE_COL, {"threshold": 0.85}, [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    (fuzzy, "first", SINGLE_COL, {"threshold": 0.75}, [0, 1, 2, 2, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    (fuzzy, "first", SINGLE_COL, {"threshold": 0.65}, [0, 1, 2, 2, 4, 5, 6, 1, 0, 0, 10, 11, 12]),
    (fuzzy, "first", SINGLE_COL, {"threshold": 0.55}, [0, 1, 2, 2, 1, 5, 2, 1, 0, 0, 1, 11, 12]),
    (fuzzy, "first", SINGLE_COL, {"threshold": 0.45}, [0, 1, 2, 2, 1, 1, 2, 1, 0, 0, 1, 0, 12]),
    (fuzzy, "first", SINGLE_COL, {"threshold": 0.35}, [0, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 12]),
    (fuzzy, "first", SINGLE_COL, {"threshold": 0.25}, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    # progressive deduping: "last"
    (fuzzy, "last", SINGLE_COL, {"threshold": 0.95}, [9, 1, 2, 3, 4, 5, 6, 7, 9, 9, 10, 11, 12]),
    (fuzzy, "last", SINGLE_COL, {"threshold": 0.85}, [9, 1, 2, 3, 4, 5, 6, 7, 9, 9, 10, 11, 12]),
    (fuzzy, "last", SINGLE_COL, {"threshold": 0.75}, [9, 1, 3, 3, 4, 5, 6, 7, 9, 9, 10, 11, 12]),
    (fuzzy, "last", SINGLE_COL, {"threshold": 0.65}, [9, 7, 3, 3, 4, 5, 6, 7, 9, 9, 10, 11, 12]),
    (fuzzy, "last", SINGLE_COL, {"threshold": 0.55}, [9, 10, 6, 6, 10, 5, 6, 10, 9, 9, 10, 11, 12]),
    (fuzzy, "last", SINGLE_COL, {"threshold": 0.45}, [11, 10, 6, 6, 10, 10, 6, 10, 11, 11, 10, 11, 12]),
    (fuzzy, "last", SINGLE_COL, {"threshold": 0.35}, [11, 11, 6, 6, 11, 11, 6, 11, 11, 11, 11, 11, 12]),
    (fuzzy, "last", SINGLE_COL, {"threshold": 0.25}, [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]),
    #
    # COSINE:
    # progressive deduping: "first"
    (cosine, "first", NUMERICAL_COMPOUND_COL, {"threshold": 0.999}, [0, 0, 0, 3, 4, 0, 0, 4, 8, 0, 0, 0, 0]),
    (cosine, "first", NUMERICAL_COMPOUND_COL, {"threshold": 0.99}, [0, 0, 0, 0, 4, 0, 0, 4, 8, 0, 0, 0, 0]),
    (cosine, "first", NUMERICAL_COMPOUND_COL, {"threshold": 0.98}, [0, 0, 0, 0, 4, 0, 0, 4, 4, 0, 0, 0, 0]),
    (cosine, "first", NUMERICAL_COMPOUND_COL, {"threshold": 0.95}, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    # progressive deduping: "last"
    (cosine, "last", NUMERICAL_COMPOUND_COL, {"threshold": 0.999}, [12, 12, 12, 3, 7, 12, 12, 7, 8, 12, 12, 12, 12]),
    (cosine, "last", NUMERICAL_COMPOUND_COL, {"threshold": 0.99}, [12, 12, 12, 12, 7, 12, 12, 7, 8, 12, 12, 12, 12]),
    (cosine, "last", NUMERICAL_COMPOUND_COL, {"threshold": 0.98}, [12, 12, 12, 12, 8, 12, 12, 8, 8, 12, 12, 12, 12]),
    (cosine, "last", NUMERICAL_COMPOUND_COL, {"threshold": 0.95}, [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]),
    #
    # JACCARD:
    # progressive deduping, "first"
    (jaccard, "first", CATEGORICAL_COMPOUND_COL, {"threshold": 0.65}, [0, 0, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 12]),
    (jaccard, "first", CATEGORICAL_COMPOUND_COL, {"threshold": 0.35}, [0, 0, 2, 3, 4, 4, 0, 3, 8, 9, 0, 0, 4]),
    (jaccard, "first", CATEGORICAL_COMPOUND_COL, {"threshold": 0.15}, [0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0]),
    (jaccard, "first", CATEGORICAL_COMPOUND_COL, {"threshold": 0.05}, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    # progressive deduping, "last"
    (jaccard, "last", CATEGORICAL_COMPOUND_COL, {"threshold": 0.65}, [6, 6, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (jaccard, "last", CATEGORICAL_COMPOUND_COL, {"threshold": 0.35}, [11, 11, 2, 7, 12, 12, 11, 7, 8, 9, 11, 11, 12]),
    (jaccard, "last", CATEGORICAL_COMPOUND_COL, {"threshold": 0.15}, [12, 12, 12, 12, 12, 12, 12, 12, 12, 9, 12, 12, 12]),
    (jaccard, "last", CATEGORICAL_COMPOUND_COL, {"threshold": 0.05}, [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]),
    #
    # lsh:
    # progressive deduping: fix ngram; vary threshold; "first"
    (lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.95, "num_perm": 128}, [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    (lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.85, "num_perm": 128}, [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    (lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.75, "num_perm": 128}, [0, 1, 2, 3, 4, 5, 6, 1, 0, 0, 10, 11, 12]),
    (lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.65, "num_perm": 128}, [0, 1, 2, 2, 4, 5, 6, 1, 0, 0, 10, 11, 12]),
    (lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 128}, [0, 1, 2, 2, 4, 5, 2, 1, 0, 0, 1, 11, 12]),
    (lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.45, "num_perm": 128},  [0, 1, 2, 2, 1, 5, 2, 1, 0, 0, 1, 0, 1]),
    (lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.35, "num_perm": 128},  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    # progressive deduping: fix threshold; vary ngram; "first"
    (lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.45, "num_perm": 128},  [0, 1, 2, 2, 1, 5, 2, 1, 0, 0, 1, 0, 1]),
    (lsh, "first", SINGLE_COL, {"ngram": 2, "threshold": 0.45, "num_perm": 128},  [0, 1, 2, 2, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    (lsh, "first", SINGLE_COL, {"ngram": 3, "threshold": 0.45, "num_perm": 128},  [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    # progressive deduping: fix parameters; vary permutations; "first"
    (lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 32}, [0, 1, 2, 2, 1, 1, 6, 1, 0, 0, 1, 11, 12]),
    (lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 64}, [0, 1, 2, 2, 1, 5, 6, 1, 0, 0, 1, 0, 1]),
    (lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 128}, [0, 1, 2, 2, 4, 5, 2, 1, 0, 0, 1, 11, 12]),
    (lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 256}, [0, 1, 2, 2, 4, 5, 2, 1, 0, 0, 1, 11, 12]),  
    # progressive deduping: fix ngram; vary threshold; "last"
    (lsh, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.95, "num_perm": 128}, [9, 1, 2, 3, 4, 5, 6, 7, 9, 9, 10, 11, 12]),
    (lsh, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.85, "num_perm": 128}, [9, 1, 2, 3, 4, 5, 6, 7, 9, 9, 10, 11, 12]),
    (lsh, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.75, "num_perm": 128}, [9, 7, 2, 3, 4, 5, 6, 7, 9, 9, 10, 11, 12]),
    (lsh, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.65, "num_perm": 128}, [9, 7, 3, 3, 4, 5, 6, 7, 9, 9, 10, 11, 12]),
    (lsh, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 128}, [9, 10, 6, 6, 4, 5, 6, 10, 9, 9, 10, 11, 12]),
    (lsh, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.45, "num_perm": 128},  [11, 12, 6, 6, 12, 5, 6, 12, 11, 11, 12, 11, 12]),
    (lsh, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.35, "num_perm": 128},  [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]),
    # progressive deduping: fix threshold; vary ngram; "last"
    (lsh, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.45, "num_perm": 128},  [11, 12, 6, 6, 12, 5, 6, 12, 11, 11, 12, 11, 12]),
    (lsh, "last", SINGLE_COL, {"ngram": 2, "threshold": 0.45, "num_perm": 128},  [9, 1, 3, 3, 4, 5, 6, 7, 9, 9, 10, 11, 12]),
    (lsh, "last", SINGLE_COL, {"ngram": 3, "threshold": 0.45, "num_perm": 128},  [9, 1, 2, 3, 4, 5, 6, 7, 9, 9, 10, 11, 12]),
    # progressive deduping: fix parameters; vary permutations; "last"
    (lsh, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 32}, [9, 10, 3, 3, 10, 10, 6, 10, 9, 9, 10, 11, 12]),
    (lsh, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 64}, [11, 12, 3, 3, 12, 5, 6, 12, 11, 11, 12, 11, 12]),
    (lsh, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 128}, [9, 10, 6, 6, 4, 5, 6, 10, 9, 9, 10, 11, 12]),
    (lsh, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 256}, [9, 10, 6, 6, 4, 5, 6, 10, 9, 9, 10, 11, 12]),    
    #
    # STRING STARTS WITH:
    # i.e. no deduping because no string starts with the pattern
    (str_startswith, "first", SINGLE_COL, {"pattern": "zzzzz", "case": True}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (str_startswith, "first", SINGLE_COL, {"pattern": "zzzzz", "case": False}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    # String does canonicalize if case correct; but doesn't otherwise
    (str_startswith, "first", SINGLE_COL, {"pattern": "calle", "case": True}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (str_startswith, "first", SINGLE_COL, {"pattern": "calle", "case": False}, [0, 1, 2, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (str_startswith, "last", SINGLE_COL, {"pattern": "calle", "case": True}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (str_startswith, "last", SINGLE_COL, {"pattern": "calle", "case": False}, [0, 1, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    #
    # STRING ENDS WITH:
    # i.e. no deduping because no string starts with the pattern
    (str_endswith, "first", SINGLE_COL, {"pattern": "zzzzz", "case": True}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (str_endswith, "first", SINGLE_COL, {"pattern": "zzzzz", "case": False}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    # String does canonicalize if case correct; but doesn't otherwise
    (str_endswith, "first", SINGLE_COL, {"pattern": "kingdom", "case": True}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (str_endswith, "first", SINGLE_COL, {"pattern": "kingdom", "case": False}, [0, 1, 2, 3, 1, 1, 6, 7, 8, 9, 10, 11, 12]),
    (str_endswith, "last", SINGLE_COL, {"pattern": "kingdom", "case": True}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (str_endswith, "last", SINGLE_COL, {"pattern": "kingdom", "case": False}, [0, 5, 2, 3, 5, 5, 6, 7, 8, 9, 10, 11, 12]),
    #
    # STRING CONTAINS:
    # i.e. no deduping because no string starts with the pattern
    (str_contains, "first", SINGLE_COL, {"pattern": "zzzzz", "case": True, "regex": True}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (str_contains, "first", SINGLE_COL, {"pattern": "zzzzz", "case": False, "regex": True}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (str_contains, "first", SINGLE_COL, {"pattern": "zzzzz", "case": True, "regex": False}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (str_contains, "first", SINGLE_COL, {"pattern": "zzzzz", "case": False, "regex": False}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    # String does canonicalize if case correct; but doesn't otherwise, no regex
    (str_contains, "first", SINGLE_COL, {"pattern": "ol5 9pl", "case": True, "regex": False}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (str_contains, "first", SINGLE_COL, {"pattern": "ol5 9pl", "case": False, "regex": False}, [0, 1, 2, 3, 4, 0, 6, 7, 0, 0, 10, 11, 12]),
    (str_contains, "last", SINGLE_COL, {"pattern": "ol5 9pl", "case": True, "regex": False}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (str_contains, "last", SINGLE_COL, {"pattern": "ol5 9pl", "case": False, "regex": False}, [9, 1, 2, 3, 4, 9, 6, 7, 9, 9, 10, 11, 12]),
    # String does canonicalize if case correct; but doesn't otherwise, with regex
    (str_contains, "first", SINGLE_COL, {"pattern": r"05\d{3}", "case": True, "regex": True}, [0, 1, 2, 2, 4, 5, 2, 7, 8, 9, 10, 11, 12]),
    (str_contains, "first", SINGLE_COL, {"pattern": r"05\d{3}", "case": False, "regex": True}, [0, 1, 2, 2, 4, 5, 2, 7, 8, 9, 10, 11, 12]),
    (str_contains, "last", SINGLE_COL, {"pattern": r"05\d{3}", "case": True, "regex": True}, [0, 1, 6, 6, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (str_contains, "last", SINGLE_COL, {"pattern": r"05\d{3}", "case": False, "regex": True}, [0, 1, 6, 6, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    #
    # TF IDF:
    # progressive deduping: vary threshold
    (tfidf, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.95, "topn": 2}, [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    (tfidf, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 2}, [0, 1, 2, 2, 4, 1, 2, 1, 0, 0, 4, 11, 12]),
    (tfidf, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.65, "topn": 2}, [0, 1, 2, 2, 4, 1, 2, 1, 0, 0, 4, 11, 1]),
    (tfidf, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.50, "topn": 2}, [0, 1, 2, 2, 4, 1, 2, 1, 0, 0, 4, 4, 1]),
    # progressive deduping: vary ngram
    (tfidf, "first", SINGLE_COL, {"ngram": (1, 2), "threshold": 0.80, "topn": 2}, [0, 1, 2, 2, 4, 5, 6, 1, 0, 0, 10, 11, 12]),
    (tfidf, "first", SINGLE_COL, {"ngram": (1, 3), "threshold": 0.80, "topn": 2}, [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    (tfidf, "first", SINGLE_COL, {"ngram": (2, 3), "threshold": 0.80, "topn": 2}, [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    # progressive deduping: vary topn
    (tfidf, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 1}, [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 10, 11, 12]),
    (tfidf, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 2}, [0, 1, 2, 2, 4, 1, 2, 1, 0, 0, 4, 11, 12]),
    (tfidf, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 3}, [0, 1, 2, 2, 1, 1, 2, 1, 0, 0, 1, 11, 12]),
    # progressive deduping: vary threshold
    (tfidf, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.95, "topn": 2}, [9, 1, 2, 3, 4, 5, 6, 7, 9, 9, 10, 11, 12]),
    (tfidf, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 2}, [9, 7, 6, 6, 10, 7, 6, 7, 9, 9, 10, 11, 12]),
    (tfidf, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.65, "topn": 2}, [9, 12, 6, 6, 10, 12, 6, 12, 9, 9, 10, 11, 12]),
    (tfidf, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.50, "topn": 2}, [9, 12, 6, 6, 11, 12, 6, 12, 9, 9, 11, 11, 12]),
    # progressive deduping: vary ngram
    (tfidf, "last", SINGLE_COL, {"ngram": (1, 2), "threshold": 0.80, "topn": 2}, [9, 7, 3, 3, 4, 5, 6, 7, 9, 9, 10, 11, 12]),
    (tfidf, "last", SINGLE_COL, {"ngram": (1, 3), "threshold": 0.80, "topn": 2}, [9, 1, 2, 3, 4, 5, 6, 7, 9, 9, 10, 11, 12]),
    (tfidf, "last", SINGLE_COL, {"ngram": (2, 3), "threshold": 0.80, "topn": 2}, [9, 1, 2, 3, 4, 5, 6, 7, 9, 9, 10, 11, 12]),
    # progressive deduping: vary topn
    (tfidf, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 1}, [9, 1, 2, 3, 4, 5, 6, 7, 9, 9, 10, 11, 12]),
    (tfidf, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 2}, [9, 7, 6, 6, 10, 7, 6, 7, 9, 9, 10, 11, 12]),
    (tfidf, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 3}, [9, 10, 6, 6, 10, 10, 6, 10, 9, 9, 10, 11, 12]),
]

# fmt: on


@pytest.mark.parametrize("strategy, rule, columns, input_params, expected_canonical_id", PARAMS)
def test_canonicalize_matrix(strategy, rule, columns, input_params, expected_canonical_id, dataframe, helpers):

    df, spark_session = dataframe

    dg = Duped(df, spark_session=spark_session, keep=rule)

    # single strategy item addition
    dg.apply(strategy(**input_params))
    dg.canonicalize(columns)

    assert helpers.get_column_as_list(dg.df, CANONICAL_ID) == expected_canonical_id

    dg = Duped(df, spark_session=spark_session, keep=rule)

    # dictionary strategy addition
    dg.apply({columns: [strategy(**input_params)]})
    dg.canonicalize()

    assert helpers.get_column_as_list(dg.df, CANONICAL_ID) == expected_canonical_id
