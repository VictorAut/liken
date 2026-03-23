"""Integration tests for output when dropping or not"""

from __future__ import annotations

import typing

import pytest

import liken as lk
from liken._constants import CANONICAL_ID


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


def simple_api(df, spark_session, columns, deduper, deduper_kwarg, drop_kwarg):
    return (
        lk.dedupe(df, spark_session=spark_session)
        .apply(deduper(**deduper_kwarg))
        .canonicalize(columns, **drop_kwarg)
        .collect()
    )


def dict_api(df, spark_session, columns, deduper, deduper_kwarg, drop_kwarg):

    return (
        lk.dedupe(df, spark_session=spark_session)
        .apply({columns: [deduper(**deduper_kwarg)]})
        .canonicalize(**drop_kwarg)
        .collect()
    )


def pipeline_api(df, spark_session, columns, deduper, deduper_kwarg, drop_kwarg):
    pipeline = lk.rules.pipeline().step(
        getattr(lk.rules.on(columns), deduper.__name__)(**deduper_kwarg)
    )
    return (
        lk.dedupe(df, spark_session=spark_session)
        .apply(pipeline)
        .canonicalize(**drop_kwarg)
        .collect()
    )


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
    (strings_same_len, "email", {"drop_duplicates": False}, {"min_len": 3}, [0, 1, 2, 3, 2, 2, 6, 3, 8, 9]),
    (strings_same_len, "email", {"drop_duplicates": True}, {"min_len": 3}, [0, 1, 2, 3, 6, 8, 9]),
    # EXACT:
    # on single column
    (lk.exact, SINGLE_COL, {"drop_duplicates": False}, {}, [0, 1, 2, 3, 4, 5, 6, 0, 4, 9]),
    (lk.exact, SINGLE_COL, {"drop_duplicates": True}, {}, [0, 1, 2, 3, 4, 5, 6, 9]),
    # on compound column
    (lk.exact, CATEGORICAL_COMPOUND_COL, {"drop_duplicates": False}, {}, [0, 0, 2, 3, 4, 5, 6, 7, 8, 9]),
    (lk.exact, CATEGORICAL_COMPOUND_COL, {"drop_duplicates": True}, {}, [0, 2, 3, 4, 5, 6, 7, 8, 9]),
    #
    # FUZZY:
    (lk.fuzzy, SINGLE_COL, {"drop_duplicates": False}, {"threshold": 0.65}, [0, 1, 2, 2, 4, 5, 1, 0, 4, 9]),
    (lk.fuzzy, SINGLE_COL, {"drop_duplicates": True}, {"threshold": 0.65}, [0, 1, 2, 4, 5, 9]),
    #
    # COSINE:
    (lk.cosine, NUMERICAL_COMPOUND_COL, {"drop_duplicates": False}, {"threshold": 0.99}, [0, 0, 0, 0, 0, 0, 6, 7, 0, 0]),
    (lk.cosine, NUMERICAL_COMPOUND_COL, {"drop_duplicates": True}, {"threshold": 0.99},  [0, 6, 7]),
    #
    # JACCARD:
    (lk.jaccard, CATEGORICAL_COMPOUND_COL, {"drop_duplicates": False}, {"threshold": 0.35}, [0, 0, 2, 3, 0, 0, 3, 7, 0, 9]),
    (lk.jaccard, CATEGORICAL_COMPOUND_COL, {"drop_duplicates": True}, {"threshold": 0.35}, [0, 2, 3, 7, 9]),
    #
    # LSH:
    (lk.lsh, SINGLE_COL, {"drop_duplicates": False}, {"ngram": 1, "threshold": 0.65, "num_perm": 128}, [0, 1, 2, 2, 4, 5, 1, 0, 4, 9]),
    (lk.lsh, SINGLE_COL, {"drop_duplicates": True}, {"ngram": 1, "threshold": 0.65, "num_perm": 128}, [0, 1, 2, 4, 5, 9]),
    #
    # STRING STARTS WITH:
    (lk.str_startswith, SINGLE_COL, {"drop_duplicates": False}, {"pattern": "calle", "case": False}, [0, 1, 2, 2, 4, 5, 6, 7, 8, 9]),
    (lk.str_startswith, SINGLE_COL, {"drop_duplicates": True}, {"pattern": "calle", "case": False}, [0, 1, 2, 4, 5, 6, 7, 8, 9]),
    #
    # STRING ENDS WITH:
    (lk.str_endswith, SINGLE_COL, {"drop_duplicates": False}, {"pattern": "kingdom", "case": False}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 1]),
    (lk.str_endswith, SINGLE_COL, {"drop_duplicates": True}, {"pattern": "kingdom", "case": False}, [0, 1, 2, 3, 4, 5, 6, 7, 8]),
    #
    # STRING CONTAINS:
    (lk.str_contains, SINGLE_COL, {"drop_duplicates": False}, {"pattern": r"05\d{3}", "case": False, "regex": True}, [0, 1, 2, 2, 4, 2, 6, 7, 8, 9]),
    (lk.str_contains, SINGLE_COL, {"drop_duplicates": True}, {"pattern": r"05\d{3}", "case": False, "regex": True}, [0, 1, 2, 4, 6, 7, 8, 9]),
    #
    # TF IDF:
    # progressive deduping: vary threshold
    (lk.tfidf, SINGLE_COL, {"drop_duplicates": False}, {"ngram": 1, "threshold": 0.80, "topn": 2}, [0, 1, 2, 2, 4, 5, 1, 0, 4, 1]),
    (lk.tfidf, SINGLE_COL, {"drop_duplicates": True}, {"ngram": 1, "threshold": 0.80, "topn": 2}, [0, 1, 2, 4, 5]),
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


@pytest.mark.parametrize(
    "deduper, columns, drop_kwarg, deduper_kwarg, expected_canonical_id",
    PARAMS,
    ids=IDS,
)
@pytest.mark.parametrize("api_builder", API_BUILDERS)
def test_matrix_dedupers(
    deduper,
    columns,
    drop_kwarg,
    deduper_kwarg,
    expected_canonical_id,
    api_builder,
    dataframe,
    helpers,
):

    df, spark_session = dataframe

    df = api_builder(df, spark_session, columns, deduper, deduper_kwarg, drop_kwarg)

    assert helpers.get_column_as_list(df, CANONICAL_ID) == expected_canonical_id