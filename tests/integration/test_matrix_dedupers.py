"""Narrow integration tests for specific behaviour of individual dedupers"""

from __future__ import annotations

import typing

import pytest

import liken as lk
from liken.constants import CANONICAL_ID


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


# HELPERS:


def simple_api(df, columns, deduper, deduper_kwarg):
    return lk.dedupe(df).apply(deduper(**deduper_kwarg)).canonicalize(columns).collect()


def dict_api(df, columns, deduper, deduper_kwarg):

    return lk.dedupe(df).apply({columns: [deduper(**deduper_kwarg)]}).canonicalize().collect()


def pipeline_api(df, columns, deduper, deduper_kwarg):
    pipeline = lk.pipeline().step(getattr(lk.col(columns), deduper.__name__)(**deduper_kwarg))
    return lk.dedupe(df).apply(pipeline).canonicalize().collect()


API_BUILDERS = [
    simple_api,
    dict_api,
    pipeline_api,
]


# REGISTER A CUSTOM CALLABLE:


@lk.custom.register
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
    (lk.exact, SINGLE_COL, {}, [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    # on compound column
    (lk.exact, CATEGORICAL_COMPOUND_COL, {}, [0, 0, 2, 3, 4, 5, 6, 7, 8, 9]),
    #
    # FUZZY:
    (lk.fuzzy, SINGLE_COL, {"threshold": 0.95}, [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    (lk.fuzzy, SINGLE_COL, {"threshold": 0.95, "scorer": "partial_ratio"}, [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    (lk.fuzzy, SINGLE_COL, {"threshold": 0.85}, [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    (lk.fuzzy, SINGLE_COL, {"threshold": 0.75}, [0, 1, 2, 2, 4, 5, 6, 0, 4, 9]),
    (lk.fuzzy, SINGLE_COL, {"threshold": 0.65}, [0, 1, 2, 2, 4, 5, 1, 0, 4, 9]),
    (lk.fuzzy, SINGLE_COL, {"threshold": 0.55}, [0, 1, 2, 2, 4, 2, 1, 0, 4, 9]),
    (lk.fuzzy, SINGLE_COL, {"threshold": 0.45}, [0, 1, 2, 2, 4, 2, 1, 0, 4, 1]),
    (lk.fuzzy, SINGLE_COL, {"threshold": 0.35}, [0, 0, 2, 2, 4, 2, 0, 0, 4, 0]),
    (lk.fuzzy, SINGLE_COL, {"threshold": 0.25}, [0, 0, 0, 0, 4, 0, 0, 0, 4, 0]),
    # jaro_winkler: pairwise similarities on this fixture are
    # 0-7 = 1.0, 2-3 = 0.905, 3-5 = 0.639, 1-6 = 0.675, 2-5 = 0.668,
    # with all remaining pairs below 0.55
    (lk.fuzzy, SINGLE_COL, {"threshold": 0.95, "scorer": "jaro_winkler"}, [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    (lk.fuzzy, SINGLE_COL, {"threshold": 0.90, "scorer": "jaro_winkler"}, [0, 1, 2, 2, 4, 5, 6, 0, 4, 9]),
    (lk.fuzzy, SINGLE_COL, {"threshold": 0.65, "scorer": "jaro_winkler"}, [0, 1, 2, 2, 4, 2, 1, 0, 4, 9]),
    (lk.fuzzy, SINGLE_COL, {"threshold": 0.55, "scorer": "jaro_winkler"}, [0, 1, 2, 2, 4, 2, 1, 0, 4, 1]),
    (lk.fuzzy, SINGLE_COL, {"threshold": 0.45, "scorer": "jaro_winkler"}, [0, 0, 0, 0, 4, 0, 0, 0, 4, 0]),
    #
    # EDIT DISTANCE:
    (lk.edit_distance, SINGLE_COL, {"max_distance": 2}, [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    (lk.edit_distance, "email", {"max_distance": 1}, [0, 1, 2, 3, 2, 2, 3, 3, 8, 0]),
    (lk.edit_distance, "email", {"max_distance": 2}, [0, 1, 0, 3, 0, 0, 3, 3, 8, 0]),
    # metric variants: normalized max_distance in [0, 1]
    (lk.edit_distance, SINGLE_COL, {"max_distance": 0.2, "metric": "damerau_levenshtein"}, [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    (lk.edit_distance, SINGLE_COL, {"max_distance": 0.2, "metric": "jaro_winkler"}, [0, 1, 2, 2, 4, 5, 6, 0, 4, 9]),
    (lk.edit_distance, SINGLE_COL, {"max_distance": 0.35, "metric": "jaro_winkler"}, [0, 1, 2, 2, 4, 2, 1, 0, 4, 9]),
    (lk.edit_distance, SINGLE_COL, {"max_distance": 0.2, "metric": "indel"}, [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    (lk.edit_distance, SINGLE_COL, {"max_distance": 0.2, "metric": "hamming"}, [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    #
    # COSINE:
    (lk.cosine, NUMERICAL_COMPOUND_COL, {"threshold": 0.999}, [0, 0, 0, 3, 0, 0, 6, 7, 0, 0]),
    (lk.cosine, NUMERICAL_COMPOUND_COL, {"threshold": 0.99}, [0, 0, 0, 0, 0, 0, 6, 7, 0, 0]),
    (lk.cosine, NUMERICAL_COMPOUND_COL, {"threshold": 0.98}, [0, 0, 0, 0, 0, 0, 6, 6, 0, 0]),
    #
    # JACCARD:
    (lk.jaccard, CATEGORICAL_COMPOUND_COL, {"threshold": 0.65}, [0, 0, 2, 3, 4, 0, 6, 7, 8, 9]),
    (lk.jaccard, CATEGORICAL_COMPOUND_COL, {"threshold": 0.35}, [0, 0, 2, 3, 0, 0, 3, 7, 0, 9]),
    #
    # TOKEN_OVERLAP:
    (lk.token_overlap, CATEGORICAL_COMPOUND_COL, {"threshold": 0.75}, [0, 0, 2, 3, 4, 0, 6, 7, 8, 9]),
    (lk.token_overlap, CATEGORICAL_COMPOUND_COL, {"threshold": 0.6}, [0, 0, 2, 3, 0, 0, 3, 7, 0, 9]),
    #
    # LSH:
    # progressive deduping: fix ngram; vary threshold
    (lk.lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.95, "num_perm": 128}, [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    (lk.lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.85, "num_perm": 128}, [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    (lk.lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.75, "num_perm": 128}, [0, 1, 2, 3, 4, 5, 1, 0, 4, 9]),
    (lk.lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.65, "num_perm": 128}, [0, 1, 2, 2, 4, 5, 1, 0, 4, 9]),
    (lk.lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 128}, [0, 1, 2, 2, 4, 2, 1, 0, 4, 9]),
    (lk.lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.45, "num_perm": 128}, [0, 1, 2, 2, 4, 2, 1, 0, 4, 9]),
    (lk.lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.35, "num_perm": 128}, [0, 1, 1, 1, 4, 1, 1, 0, 4, 0]),
    # progressive deduping: fix threshold; vary ngram
    (lk.lsh, SINGLE_COL, {"ngram": 2, "threshold": 0.45, "num_perm": 128}, [0, 1, 2, 2, 4, 5, 6, 0, 4, 9]),
    (lk.lsh, SINGLE_COL, {"ngram": 3, "threshold": 0.45, "num_perm": 128}, [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    # progressive deduping: fix parameters; vary permutations
    (lk.lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 32}, [0, 1, 2, 2, 4, 5, 1, 0, 4, 9]),
    (lk.lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 64}, [0, 1, 2, 2, 4, 5, 1, 0, 4, 9]),
    (lk.lsh, SINGLE_COL, {"ngram": 1, "threshold": 0.55, "num_perm": 128}, [0, 1, 2, 2, 4, 2, 1, 0, 4, 9]),
    #
    # STRING STARTS WITH:
    # i.e. no deduping because no string starts with the pattern
    (lk.str_startswith, SINGLE_COL, {"pattern": "zzzzz", "case": True}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    (lk.str_startswith, SINGLE_COL, {"pattern": "zzzzz", "case": False}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    # String does canonicalize if case correct; but doesn't otherwise
    (lk.str_startswith, SINGLE_COL, {"pattern": "calle", "case": True}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    (lk.str_startswith, SINGLE_COL, {"pattern": "calle", "case": False}, [0, 1, 2, 2, 4, 5, 6, 7, 8, 9]),
    #
    # STRING ENDS WITH:
    # i.e. no deduping because no string starts with the pattern
    (lk.str_endswith, SINGLE_COL, {"pattern": "zzzzz", "case": True}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    (lk.str_endswith, SINGLE_COL, {"pattern": "zzzzz", "case": False}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    # String does canonicalize if case correct; but doesn't otherwise
    (lk.str_endswith, SINGLE_COL, {"pattern": "kingdom", "case": True}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    (lk.str_endswith, SINGLE_COL, {"pattern": "kingdom", "case": False}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 1]),
    #
    # STRING LEN:
    # no deduping because bounds are out of range
    (lk.str_len, "email", {"min_len": 101, "max_len": 201}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    # total deduping given no bounds
    (lk.str_len, "email", {}, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (lk.str_len, "email", {"min_len": 16, "max_len": 22}, [0, 1, 2, 0, 4, 5, 0, 0, 8, 9]),
    # exact-length idiom
    (lk.str_len, "email", {"min_len": 22, "max_len": 22}, [0, 1, 2, 3, 4, 5, 6, 3, 8, 9]),
    #
    # STRING CONTAINS:
    # i.e. no deduping because no string starts with the pattern
    (lk.str_contains, SINGLE_COL, {"pattern": "zzzzz", "case": True, "regex": True}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    (lk.str_contains, SINGLE_COL, {"pattern": "zzzzz", "case": False, "regex": True}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    (lk.str_contains, SINGLE_COL, {"pattern": "zzzzz", "case": True, "regex": False}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    (lk.str_contains, SINGLE_COL, {"pattern": "zzzzz", "case": False, "regex": False}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    # String does canonicalize if case correct; but doesn't otherwise, no regex
    (lk.str_contains, SINGLE_COL, {"pattern": "ol5 9pl", "case": True, "regex": False}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    (lk.str_contains, SINGLE_COL, {"pattern": "ol5 9pl", "case": False, "regex": False}, [0, 1, 2, 3, 4, 5, 6, 0, 8, 0]),
    # String does canonicalize if case correct; but doesn't otherwise, with regex
    (lk.str_contains, SINGLE_COL, {"pattern": r"05\d{3}", "case": True, "regex": True}, [0, 1, 2, 2, 4, 2, 6, 7, 8, 9]),
    (lk.str_contains, SINGLE_COL, {"pattern": r"05\d{3}", "case": False, "regex": True}, [0, 1, 2, 2, 4, 2, 6, 7, 8, 9]),
    #
    # TF IDF:
    # progressive deduping: vary threshold
    (lk.tfidf, SINGLE_COL, {"ngram": 1, "threshold": 0.95, "topn": 2}, [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    (lk.tfidf, SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 2}, [0, 1, 2, 2, 4, 5, 1, 0, 4, 1]),
    (lk.tfidf, SINGLE_COL, {"ngram": 1, "threshold": 0.65, "topn": 2}, [0, 1, 2, 2, 4, 2, 1, 0, 4, 1]),
    (lk.tfidf, SINGLE_COL, {"ngram": 1, "threshold": 0.50, "topn": 2}, [0, 1, 2, 2, 4, 2, 1, 0, 4, 1]),
    # progressive deduping: vary ngram
    (lk.tfidf, SINGLE_COL, {"ngram": (1, 2), "threshold": 0.80, "topn": 2}, [0, 1, 2, 2, 4, 5, 1, 0, 4, 9]),
    (lk.tfidf, SINGLE_COL, {"ngram": (1, 3), "threshold": 0.80, "topn": 2}, [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    (lk.tfidf, SINGLE_COL, {"ngram": (2, 3), "threshold": 0.80, "topn": 2}, [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    # progressive deduping: vary topn
    (lk.tfidf, SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 1}, [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    (lk.tfidf, SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 2}, [0, 1, 2, 2, 4, 5, 1, 0, 4, 1]),
    (lk.tfidf, SINGLE_COL, {"ngram": 1, "threshold": 0.80, "topn": 3}, [0, 1, 2, 2, 4, 5, 1, 0, 4, 1]),
    #
    # ISIN:
    # rows 1 and 8 share the address "123ab, OL5 9PL, UK"
    (lk.isin, SINGLE_COL, {"values": ["123ab, OL5 9PL, UK"]}, [0, 1, 2, 3, 4, 5, 6, 0, 8, 9]),
    # two distinct addresses matched, merging rows 3 and 4
    (
        lk.isin,
        SINGLE_COL,
        {"values": ["Calle Ancho, 12, 05688, Rioja, Navarra, Espana", "Calle Sueco, 56, 05688, Rioja, Navarra"]},
        [0, 1, 2, 2, 4, 5, 6, 7, 8, 9],
    ),
    # no address in the given set, so no deduping
    (lk.isin, SINGLE_COL, {"values": ["zzzzz"]}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    #
    # ISNA:
    (lk.isna, SINGLE_COL, {}, [0, 1, 2, 3, 4, 5, 6, 7, 4, 9]),
    #
    # NUM RANGE:
    (lk.num_range, "property_area_sq_ft", {"min": 500, "max": 620}, [0, 1, 2, 3, 4, 5, 0, 0, 8, 9]),
    # min-only:
    (lk.num_range, "property_area_sq_ft", {"min": 600}, [0, 1, 2, 2, 2, 2, 6, 7, 8, 2]),
    # max-only:
    (lk.num_range, "property_area_sq_ft", {"max": 600}, [0, 0, 2, 3, 4, 5, 0, 0, 0, 9]),
    # no value in range: no deduping
    (lk.num_range, "property_area_sq_ft", {"min": 5000, "max": 6000}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    #
    # WITHIN:
    # tol=0: numeric equality, all values distinct
    (lk.within, "property_area_sq_ft", {"tol": 0}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    # 545-500 = 45 exactly: the inclusive bound merges {500, 509, 545}
    (lk.within, "property_area_sq_ft", {"tol": 45}, [0, 1, 2, 3, 4, 5, 0, 0, 8, 9]),
    # transitive chain 452-500 (48), 500-545 (45), 545-623 (78) merges all
    # five, though 452 and 623 are 171 apart
    (lk.within, "property_area_sq_ft", {"tol": 100}, [0, 0, 0, 3, 4, 5, 0, 0, 8, 9]),
    # wide tolerance collapses the whole column (skew shape)
    (lk.within, "property_area_sq_ft", {"tol": 2000}, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    # property_height carries nulls (rows 0 and 8): nulls group together,
    # the 2.5s merge, and 2.7 joins at tol=0.2 — its float difference
    # 0.20000000000000018 rides the 1e-9 boundary guard
    (lk.within, "property_height", {"tol": 0.1}, [0, 0, 2, 3, 4, 2, 2, 2, 2, 2]),
    (lk.within, "property_height", {"tol": 0.2}, [0, 0, 2, 3, 2, 2, 2, 2, 2, 2]),
    #
    # DATE RANGE:
    # date_joined: rows 0,1 = Jan 5; row 2 = Jan 7; row 3 = Jan 8;
    # rows 4,8 = null; row 5 = Jun 1; rows 6,7 = Mar 15; row 9 = Jun 2.
    # Expectations derived by brute force over the dates, then confirmed
    # by run.
    # window=0 is same-date matching
    (lk.date_range, "date_joined", {"format": "%Y-%m-%d"}, [0, 0, 2, 3, 4, 5, 6, 6, 4, 9]),
    # window=1: Jan 7-Jan 8 and Jun 1-Jun 2 merge; Jan 5 stays apart from
    # Jan 7 (2 days)
    (lk.date_range, "date_joined", {"format": "%Y-%m-%d", "window": 1}, [0, 0, 2, 2, 4, 5, 6, 6, 4, 5]),
    # window=2: Jan 5, Jan 7 and Jan 8 merge — the inclusive bound at
    # exactly 2 days crosses the plain date-key boundary
    (lk.date_range, "date_joined", {"format": "%Y-%m-%d", "window": 2}, [0, 0, 0, 0, 4, 5, 6, 6, 4, 5]),
    #
    # SORTED TOKENS:
    # name column: rows 0, 1, 8 share the tokens {john, smith} in any
    # order; rows 2, 3, 6 share {ann, lee, mary}; row 4 is "" and row 9 is
    # whitespace-only — zero-token values match nothing on the positive
    # path; row 5 is null.
    (lk.sorted_tokens, "name", {}, [0, 0, 2, 2, 4, 5, 2, 7, 0, 9]),
]

# fmt: on


@pytest.mark.parametrize("deduper, columns, dedup_kwarg, expected_canonical_id", PARAMS)
@pytest.mark.parametrize("api_builder", API_BUILDERS)
def test_matrix_dedupers(
    deduper,
    columns,
    dedup_kwarg,
    expected_canonical_id,
    api_builder,
    dataframe,
    helpers,
):

    df = api_builder(
        df=dataframe,
        columns=columns,
        deduper=deduper,
        deduper_kwarg=dedup_kwarg,
    )

    assert helpers.get_column_as_list(df, CANONICAL_ID) == expected_canonical_id
