"""Narrow integration tests for specific behaviour of individual stratgies"""

from __future__ import annotations
import typing

import pytest

from dupegrouper import Duped
from dupegrouper.constants import CANONICAL_ID
from dupegrouper.strats_library import (
    cosine,
    exact,
    fuzzy,
    jaccard,
    lsh,
    str_startswith,
    str_endswith,
    str_contains,
    tfidf,
)


# CONSTANTS:


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


# CUSTOM CALLABLE:


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
    # EXACT:
    # on single column
    (exact, "first", SINGLE_COL, {}, [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 11, 12, 13]),
    (exact, "last", SINGLE_COL, {}, [10, 2, 3, 4, 5, 6, 7, 8, 10, 10, 11, 12, 13]),
    # on compound column
    (exact, "first", CATEGORICAL_COMPOUND_COL, {}, [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    (exact, "last", CATEGORICAL_COMPOUND_COL, {}, [2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    #
    # FUZZY:
    # progressive deduping: "first"
    (fuzzy, "first", SINGLE_COL, {"threshold": 0.95}, [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 11, 12, 13]),
    (fuzzy, "first", SINGLE_COL, {"threshold": 0.85}, [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 11, 12, 13]),
    (fuzzy, "first", SINGLE_COL, {"threshold": 0.75}, [1, 2, 3, 3, 5, 6, 7, 8, 1, 1, 11, 12, 13]),
    (fuzzy, "first", SINGLE_COL, {"threshold": 0.65}, [1, 2, 3, 3, 5, 6, 7, 2, 1, 1, 11, 12, 13]),
    (fuzzy, "first", SINGLE_COL, {"threshold": 0.55}, [1, 2, 3, 3, 2, 6, 3, 2, 1, 1, 2, 12, 13]),
    (fuzzy, "first", SINGLE_COL, {"threshold": 0.45}, [1, 2, 3, 3, 2, 2, 3, 2, 1, 1, 2, 1, 13]),
    (fuzzy, "first", SINGLE_COL, {"threshold": 0.35}, [1, 1, 3, 3, 1, 1, 3, 1, 1, 1, 1, 1, 13]),
    (fuzzy, "first", SINGLE_COL, {"threshold": 0.25}, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    # progressive deduping: "last"
    (fuzzy, "last", SINGLE_COL, {"threshold": 0.95}, [10, 2, 3, 4, 5, 6, 7, 8, 10, 10, 11, 12, 13]),
    (fuzzy, "last", SINGLE_COL, {"threshold": 0.85}, [10, 2, 3, 4, 5, 6, 7, 8, 10, 10, 11, 12, 13]),
    (fuzzy, "last", SINGLE_COL, {"threshold": 0.75}, [10, 2, 4, 4, 5, 6, 7, 8, 10, 10, 11, 12, 13]),
    (fuzzy, "last", SINGLE_COL, {"threshold": 0.65}, [10, 8, 4, 4, 5, 6, 7, 8, 10, 10, 11, 12, 13]),
    (fuzzy, "last", SINGLE_COL, {"threshold": 0.55}, [10, 11, 7, 7, 11, 6, 7, 11, 10, 10, 11, 12, 13]),
    (fuzzy, "last", SINGLE_COL, {"threshold": 0.45}, [12, 11, 7, 7, 11, 11, 7, 11, 12, 12, 11, 12, 13]),
    (fuzzy, "last", SINGLE_COL, {"threshold": 0.35}, [12, 12, 7, 7, 12, 12, 7, 12, 12, 12, 12, 12, 13]),
    (fuzzy, "last", SINGLE_COL, {"threshold": 0.25}, [13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13]),
    #
    # COSINE:
    # progressive deduping: "first"
    (cosine, "first", NUMERICAL_COMPOUND_COL, {"threshold": 0.999}, [1, 1, 1, 4, 5, 1, 1, 5, 9, 1, 1, 1, 1]),
    (cosine, "first", NUMERICAL_COMPOUND_COL, {"threshold": 0.99}, [1, 1, 1, 1, 5, 1, 1, 5, 9, 1, 1, 1, 1]),
    (cosine, "first", NUMERICAL_COMPOUND_COL, {"threshold": 0.98}, [1, 1, 1, 1, 5, 1, 1, 5, 5, 1, 1, 1, 1]),
    (cosine, "first", NUMERICAL_COMPOUND_COL, {"threshold": 0.95}, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    # progressive deduping: "last"
    (cosine, "last", NUMERICAL_COMPOUND_COL, {"threshold": 0.999}, [13, 13, 13, 4, 8, 13, 13, 8, 9, 13, 13, 13, 13]),
    (cosine, "last", NUMERICAL_COMPOUND_COL, {"threshold": 0.99}, [13, 13, 13, 13, 8, 13, 13, 8, 9, 13, 13, 13, 13]),
    (cosine, "last", NUMERICAL_COMPOUND_COL, {"threshold": 0.98}, [13, 13, 13, 13, 9, 13, 13, 9, 9, 13, 13, 13, 13]),
    (cosine, "last", NUMERICAL_COMPOUND_COL, {"threshold": 0.95}, [13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13]),
    #
    # JACCARD:
    # progressive deduping, "first"
    (jaccard, "first", CATEGORICAL_COMPOUND_COL, {"threshold": 0.65}, [1, 1, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 13]),
    (jaccard, "first", CATEGORICAL_COMPOUND_COL, {"threshold": 0.35}, [1, 1, 3, 4, 5, 5, 1, 4, 9, 10, 1, 1, 5]),
    (jaccard, "first", CATEGORICAL_COMPOUND_COL, {"threshold": 0.15}, [1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 1, 1, 1]),
    (jaccard, "first", CATEGORICAL_COMPOUND_COL, {"threshold": 0.05}, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    # progressive deduping, "last"
    (jaccard, "last", CATEGORICAL_COMPOUND_COL, {"threshold": 0.65}, [7, 7, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    (jaccard, "last", CATEGORICAL_COMPOUND_COL, {"threshold": 0.35}, [12, 12, 3, 8, 13, 13, 12, 8, 9, 10, 12, 12, 13]),
    (jaccard, "last", CATEGORICAL_COMPOUND_COL, {"threshold": 0.15}, [13, 13, 13, 13, 13, 13, 13, 13, 13, 10, 13, 13, 13]),
    (jaccard, "last", CATEGORICAL_COMPOUND_COL, {"threshold": 0.05}, [13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13]),
    #
    # lsh:
    # progressive deduping: fix ngram; vary threshold; "first"
    (lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.95, "num_perm": 128}, [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 11, 12, 13]),
    (lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.85, "num_perm": 128}, [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 11, 12, 13]),
    (lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.75, "num_perm": 128}, [1, 2, 3, 4, 5, 6, 7, 2, 1, 1, 11, 12, 13]),
    (lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.65, "num_perm": 128}, [1, 2, 3, 3, 5, 6, 7, 2, 1, 1, 11, 12, 13]),
    (lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 128}, [1, 2, 3, 3, 5, 6, 3, 2, 1, 1, 2, 12, 13]),
    (lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.45, "num_perm": 128},  [1, 2, 3, 3, 2, 6, 3, 2, 1, 1, 2, 1, 2]),
    (lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.35, "num_perm": 128},  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    # progressive deduping: fix threshold; vary ngram; "first"
    (lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.45, "num_perm": 128},  [1, 2, 3, 3, 2, 6, 3, 2, 1, 1, 2, 1, 2]),
    (lsh, "first", SINGLE_COL, {"ngram": 2, "threshold": 0.45, "num_perm": 128},  [1, 2, 3, 3, 5, 6, 7, 8, 1, 1, 11, 12, 13]),
    (lsh, "first", SINGLE_COL, {"ngram": 3, "threshold": 0.45, "num_perm": 128},  [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 11, 12, 13]),
    # progressive deduping: fix parameters; vary permutations; "first"
    (lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 32}, [1, 2, 3, 3, 2, 2, 7, 2, 1, 1, 2, 12, 13]),
    (lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 64}, [1, 2, 3, 3, 2, 6, 7, 2, 1, 1, 2, 1, 2]),
    (lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 128}, [1, 2, 3, 3, 5, 6, 3, 2, 1, 1, 2, 12, 13]),
    (lsh, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 256}, [1, 2, 3, 3, 5, 6, 3, 2, 1, 1, 2, 12, 13]),  
    # progressive deduping: fix ngram; vary threshold; "last"
    (lsh, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.95, "num_perm": 128}, [10, 2, 3, 4, 5, 6, 7, 8, 10, 10, 11, 12, 13]),
    (lsh, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.85, "num_perm": 128}, [10, 2, 3, 4, 5, 6, 7, 8, 10, 10, 11, 12, 13]),
    (lsh, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.75, "num_perm": 128}, [10, 8, 3, 4, 5, 6, 7, 8, 10, 10, 11, 12, 13]),
    (lsh, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.65, "num_perm": 128}, [10, 8, 4, 4, 5, 6, 7, 8, 10, 10, 11, 12, 13]),
    (lsh, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 128}, [10, 11, 7, 7, 5, 6, 7, 11, 10, 10, 11, 12, 13]),
    (lsh, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.45, "num_perm": 128},  [12, 13, 7, 7, 13, 6, 7, 13, 12, 12, 13, 12, 13]),
    (lsh, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.35, "num_perm": 128},  [13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13]),
    # progressive deduping: fix threshold; vary ngram; "last"
    (lsh, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.45, "num_perm": 128},  [12, 13, 7, 7, 13, 6, 7, 13, 12, 12, 13, 12, 13]),
    (lsh, "last", SINGLE_COL, {"ngram": 2, "threshold": 0.45, "num_perm": 128},  [10, 2, 4, 4, 5, 6, 7, 8, 10, 10, 11, 12, 13]),
    (lsh, "last", SINGLE_COL, {"ngram": 3, "threshold": 0.45, "num_perm": 128},  [10, 2, 3, 4, 5, 6, 7, 8, 10, 10, 11, 12, 13]),
    # progressive deduping: fix parameters; vary permutations; "last"
    (lsh, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 32}, [10, 11, 4, 4, 11, 11, 7, 11, 10, 10, 11, 12, 13]),
    (lsh, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 64}, [12, 13, 4, 4, 13, 6, 7, 13, 12, 12, 13, 12, 13]),
    (lsh, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 128}, [10, 11, 7, 7, 5, 6, 7, 11, 10, 10, 11, 12, 13]),
    (lsh, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 256}, [10, 11, 7, 7, 5, 6, 7, 11, 10, 10, 11, 12, 13]),    
    #
    # STRING STARTS WITH:
    # i.e. no deduping because no string starts with the pattern
    (str_startswith, "first", SINGLE_COL, {"pattern": "zzzzz", "case": True}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    (str_startswith, "first", SINGLE_COL, {"pattern": "zzzzz", "case": False}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # String does canonicalize if case correct; but doesn't otherwise
    (str_startswith, "first", SINGLE_COL, {"pattern": "calle", "case": True}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    (str_startswith, "first", SINGLE_COL, {"pattern": "calle", "case": False}, [1, 2, 3, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    (str_startswith, "last", SINGLE_COL, {"pattern": "calle", "case": True}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    (str_startswith, "last", SINGLE_COL, {"pattern": "calle", "case": False}, [1, 2, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    #
    # STRING ENDS WITH:
    # i.e. no deduping because no string starts with the pattern
    (str_endswith, "first", SINGLE_COL, {"pattern": "zzzzz", "case": True}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    (str_endswith, "first", SINGLE_COL, {"pattern": "zzzzz", "case": False}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # String does canonicalize if case correct; but doesn't otherwise
    (str_endswith, "first", SINGLE_COL, {"pattern": "kingdom", "case": True}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    (str_endswith, "first", SINGLE_COL, {"pattern": "kingdom", "case": False}, [1, 2, 3, 4, 2, 2, 7, 8, 9, 10, 11, 12, 13]),
    (str_endswith, "last", SINGLE_COL, {"pattern": "kingdom", "case": True}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    (str_endswith, "last", SINGLE_COL, {"pattern": "kingdom", "case": False}, [1, 6, 3, 4, 6, 6, 7, 8, 9, 10, 11, 12, 13]),
    #
    # STRING CONTAINS:
    # i.e. no deduping because no string starts with the pattern
    (str_contains, "first", SINGLE_COL, {"pattern": "zzzzz", "case": True, "regex": True}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    (str_contains, "first", SINGLE_COL, {"pattern": "zzzzz", "case": False, "regex": True}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    (str_contains, "first", SINGLE_COL, {"pattern": "zzzzz", "case": True, "regex": False}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    (str_contains, "first", SINGLE_COL, {"pattern": "zzzzz", "case": False, "regex": False}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # String does canonicalize if case correct; but doesn't otherwise, no regex
    (str_contains, "first", SINGLE_COL, {"pattern": "ol5 9pl", "case": True, "regex": False}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    (str_contains, "first", SINGLE_COL, {"pattern": "ol5 9pl", "case": False, "regex": False}, [1, 2, 3, 4, 5, 1, 7, 8, 1, 1, 11, 12, 13]),
    (str_contains, "last", SINGLE_COL, {"pattern": "ol5 9pl", "case": True, "regex": False}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    (str_contains, "last", SINGLE_COL, {"pattern": "ol5 9pl", "case": False, "regex": False}, [10, 2, 3, 4, 5, 10, 7, 8, 10, 10, 11, 12, 13]),
    # String does canonicalize if case correct; but doesn't otherwise, with regex
    (str_contains, "first", SINGLE_COL, {"pattern": r"05\d{3}", "case": True, "regex": True}, [1, 2, 3, 3, 5, 6, 3, 8, 9, 10, 11, 12, 13]),
    (str_contains, "first", SINGLE_COL, {"pattern": r"05\d{3}", "case": False, "regex": True}, [1, 2, 3, 3, 5, 6, 3, 8, 9, 10, 11, 12, 13]),
    (str_contains, "last", SINGLE_COL, {"pattern": r"05\d{3}", "case": True, "regex": True}, [1, 2, 7, 7, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    (str_contains, "last", SINGLE_COL, {"pattern": r"05\d{3}", "case": False, "regex": True}, [1, 2, 7, 7, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    #
    # TF IDF:
    # progressive deduping: vary threshold
    (tfidf, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.95, "topn": 2}, [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 11, 12, 13]),
    (tfidf, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 2}, [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 12, 13]),
    (tfidf, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.65, "topn": 2}, [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 12, 2]),
    (tfidf, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.50, "topn": 2}, [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 5, 2]),
    # progressive deduping: vary ngram
    (tfidf, "first", SINGLE_COL, {"ngram": (1, 2), "threshold": 0.80, "topn": 2}, [1, 2, 3, 3, 5, 6, 7, 2, 1, 1, 11, 12, 13]),
    (tfidf, "first", SINGLE_COL, {"ngram": (1, 3), "threshold": 0.80, "topn": 2}, [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 11, 12, 13]),
    (tfidf, "first", SINGLE_COL, {"ngram": (2, 3), "threshold": 0.80, "topn": 2}, [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 11, 12, 13]),
    # progressive deduping: vary topn
    (tfidf, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 1}, [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 11, 12, 13]),
    (tfidf, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 2}, [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 12, 13]),
    (tfidf, "first", SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 3}, [1, 2, 3, 3, 2, 2, 3, 2, 1, 1, 2, 12, 13]),
    # progressive deduping: vary threshold
    (tfidf, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.95, "topn": 2}, [10, 2, 3, 4, 5, 6, 7, 8, 10, 10, 11, 12, 13]),
    (tfidf, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 2}, [10, 8, 7, 7, 11, 8, 7, 8, 10, 10, 11, 12, 13]),
    (tfidf, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.65, "topn": 2}, [10, 13, 7, 7, 11, 13, 7, 13, 10, 10, 11, 12, 13]),
    (tfidf, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.50, "topn": 2}, [10, 13, 7, 7, 12, 13, 7, 13, 10, 10, 12, 12, 13]),
    # progressive deduping: vary ngram
    (tfidf, "last", SINGLE_COL, {"ngram": (1, 2), "threshold": 0.80, "topn": 2}, [10, 8, 4, 4, 5, 6, 7, 8, 10, 10, 11, 12, 13]),
    (tfidf, "last", SINGLE_COL, {"ngram": (1, 3), "threshold": 0.80, "topn": 2}, [10, 2, 3, 4, 5, 6, 7, 8, 10, 10, 11, 12, 13]),
    (tfidf, "last", SINGLE_COL, {"ngram": (2, 3), "threshold": 0.80, "topn": 2}, [10, 2, 3, 4, 5, 6, 7, 8, 10, 10, 11, 12, 13]),
    # progressive deduping: vary topn
    (tfidf, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 1}, [10, 2, 3, 4, 5, 6, 7, 8, 10, 10, 11, 12, 13]),
    (tfidf, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 2}, [10, 8, 7, 7, 11, 8, 7, 8, 10, 10, 11, 12, 13]),
    (tfidf, "last", SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 3}, [10, 11, 7, 7, 11, 11, 7, 11, 10, 10, 11, 12, 13]),
]

# fmt: on


@pytest.mark.parametrize("strategy, rule, columns, input_params, expected_canonical_id", PARAMS)
def test_canonicalize_matrix(strategy, rule, columns, input_params, expected_canonical_id, dataframe, helpers):

    df, spark_session, id = dataframe

    dg = Duped(df, spark_session=spark_session, id=id, keep=rule)

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
