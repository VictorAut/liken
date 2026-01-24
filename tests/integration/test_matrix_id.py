"""Narrow integration tests for specific behaviour of individual stratgies"""

from __future__ import annotations

import warnings

import pandas as pd
import polars as pl
import pytest

from dupegrouper import Dedupe, exact
from dupegrouper._constants import CANONICAL_ID


# CONSTANTS:


SINGLE_COL = "address"


PARAMS = [
    # "CREATE"
    # creates a new auto-incremental `canonical_id`
    (
        None,
        ["uid", "address"],
        [
            [1, "123ab, OL5"],
            [2, "Westside Avenue"],
            [3, "123ab, OL5"],
        ],
        [0, 1, 0],
    ),
    # "COPY"
    # copy the defined field from the data as the `canonical_id`
    (
        "uid",
        ["uid", "address"],
        [
            [1, "123ab, OL5"],
            [2, "Westside Avenue"],
            [3, "123ab, OL5"],
        ],
        [1, 2, 1],
    ),
    (
        "uid",
        ["uid", "address"],
        [
            ["a001", "123ab, OL5"],
            ["a002", "Westside Avenue"],
            ["a003", "123ab, OL5"],
        ],
        ["a001", "a002", "a001"],
    ),
    # "OVERWRITE"
    # overwrite a pre-existing `canonical_id` field, or not
    (
        None,
        ["uid", "address", "canonical_id"],
        [
            [1, "123ab, OL5", 10],
            [2, "Westside Avenue", 12],
            [3, "123ab, OL5", 10],
        ],
        [10, 12, 10],
    ),
    # next example is good example of iterative deduping based on appended rows
    (
        None,
        ["uid", "address", "canonical_id"],
        [
            [1, "123ab, OL5", 10],
            [2, "Westside Avenue", 12],
            [3, "123ab, OL5", 13],
        ],
        [10, 12, 10],
    ),
    # same results are obtained if id is passed as `canonical_id`
    (
        "canonical_id",
        ["uid", "address", "canonical_id"],
        [
            [1, "123ab, OL5", 10],
            [2, "Westside Avenue", 12],
            [3, "123ab, OL5", 10],
        ],
        [10, 12, 10],
    ),
    (
        "canonical_id",
        ["uid", "address", "canonical_id"],
        [
            [1, "123ab, OL5", 10],
            [2, "Westside Avenue", 12],
            [3, "123ab, OL5", 13],
        ],
        [10, 12, 10],
    ),
    # finally, can actually overwrite the canonical_id
    (
        "uid",
        ["uid", "address", "canonical_id"],
        [
            [1, "123ab, OL5", 10],
            [2, "Westside Avenue", 12],
            [3, "123ab, OL5", 10],
        ],
        [1, 2, 1],
    ),
    (
        "uid",
        ["uid", "address", "canonical_id"],
        [
            ["e00005", "123ab, OL5", 10],
            ["e00006", "Westside Avenue", 12],
            ["e00009", "123ab, OL5", 10],
        ],
        ["e00005", "e00006", "e00005"],
    ),
    (
        "uid",
        ["uid", "address", "canonical_id"],
        [
            [10, "123ab, OL5", "e00005"],
            [12, "Westside Avenue", "e00006"],
            [13, "123ab, OL5", "e00009"],
        ],
        [10, 12, 10],
    ),
    (
        "uid",
        ["uid", "address", "canonical_id"],
        [
            ["e00005", "123ab, OL5", "10"],
            ["e00006", "Westside Avenue", "12"],
            ["e00009", "123ab, OL5", "10"],
        ],
        ["e00005", "e00006", "e00005"],
    ),
]
IDS = [
    "new-auto-incremental-canonical_id",
    "copy-numeric-id",
    "copy-string-id",
    "canonical_id-already-exists-not-dededupe",
    "canonical_id-already-exists-partially-dededupe",
    "canonical_id-already-exists-not-dededupe-verbose",
    "canonical_id-already-exists-partially-dededupe-verbose",
    "overwrite-numeric-to-numeric",
    "overwrite-string-to-numeric",
    "overwrite-numeric-to-string",
    "overwrite-string-to-string",
]


@pytest.mark.parametrize("id, schema, data, expected_canonical_id", PARAMS, ids=IDS)
@pytest.mark.parametrize("backend", ["pandas", "polars", "spark"])
def test_matrix_id(backend, id, schema, data, expected_canonical_id, spark, helpers):

    if backend == "pandas":
        df = pd.DataFrame(columns=schema, data=data)

    if backend == "polars":
        df = pl.DataFrame(schema=schema, data=data, orient="row")

    if backend == "spark":
        df = spark.createDataFrame(schema=schema, data=data)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        dp = Dedupe(df, spark_session=spark, id=id)

    dp.apply(exact())
    df = dp.canonicalize(SINGLE_COL)

    assert helpers.get_column_as_list(df, CANONICAL_ID) == expected_canonical_id
