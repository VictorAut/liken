"""Narrow integration tests for creation of canonical ID column"""

from __future__ import annotations

import re

import pytest

import liken as lk
from liken.constants import CANONICAL_ID
from liken.constants import INVALID_CANONICAL_ID_COLLISION_MSG


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
        CANONICAL_ID,
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
        CANONICAL_ID,
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
        CANONICAL_ID,
    ),
    # "CUSTOM LABEL"
    # a non-default output label threads through every backend
    (
        None,
        ["uid", "address"],
        [
            [1, "123ab, OL5"],
            [2, "Westside Avenue"],
            [3, "123ab, OL5"],
        ],
        [0, 1, 0],
        "master_id",
    ),
]
IDS = [
    "new-auto-incremental-canonical_id",
    "copy-numeric-id",
    "copy-string-id",
    "custom-output-label",
]

# "COLLISION"
# a frame that already carries a `canonical_id` column is rejected at entry;
# the overwrite and as-is behaviours this matrix used to express are no longer
# reachable from `canonicalize`.
COLLISION_PARAMS = [
    # as-is: no source_id, frame already carries the column
    (
        None,
        ["uid", "address", "canonical_id"],
        [
            [1, "123ab, OL5", 10],
            [2, "Westside Avenue", 12],
            [3, "123ab, OL5", 10],
        ],
    ),
    # partially deduped canonical ids on the frame
    (
        None,
        ["uid", "address", "canonical_id"],
        [
            [1, "123ab, OL5", 10],
            [2, "Westside Avenue", 12],
            [3, "123ab, OL5", 13],
        ],
    ),
    # source_id equal to the colliding label
    (
        "canonical_id",
        ["uid", "address", "canonical_id"],
        [
            [1, "123ab, OL5", 10],
            [2, "Westside Avenue", 12],
            [3, "123ab, OL5", 10],
        ],
    ),
    # overwrite attempts: source_id names a different column
    (
        "uid",
        ["uid", "address", "canonical_id"],
        [
            [1, "123ab, OL5", 10],
            [2, "Westside Avenue", 12],
            [3, "123ab, OL5", 10],
        ],
    ),
    (
        "uid",
        ["uid", "address", "canonical_id"],
        [
            ["e00005", "123ab, OL5", 10],
            ["e00006", "Westside Avenue", 12],
            ["e00009", "123ab, OL5", 10],
        ],
    ),
    (
        "uid",
        ["uid", "address", "canonical_id"],
        [
            [10, "123ab, OL5", "e00005"],
            [12, "Westside Avenue", "e00006"],
            [13, "123ab, OL5", "e00009"],
        ],
    ),
    (
        "uid",
        ["uid", "address", "canonical_id"],
        [
            ["e00005", "123ab, OL5", "10"],
            ["e00006", "Westside Avenue", "12"],
            ["e00009", "123ab, OL5", "10"],
        ],
    ),
]
COLLISION_IDS = [
    "canonical_id-already-exists-not-dededupe",
    "canonical_id-already-exists-partially-dededupe",
    "source_id-equals-colliding-label",
    "overwrite-numeric-to-numeric",
    "overwrite-string-to-numeric",
    "overwrite-numeric-to-string",
    "overwrite-string-to-string",
]


@pytest.mark.parametrize("source_id, schema, data, expected_canonical_id, output_label", PARAMS, ids=IDS)
def test_matrix_id(
    source_id,
    schema,
    data,
    expected_canonical_id,
    output_label,
    helpers,
):

    df = helpers.create_df(data, schema)

    df = (
        lk.dedupe(df)
        .apply(lk.exact())
        .canonicalize(SINGLE_COL, source_id=source_id, canonical_id=output_label)
        .collect()
    )

    assert helpers.get_column_as_list(df, output_label) == expected_canonical_id


@pytest.mark.parametrize("source_id, schema, data", COLLISION_PARAMS, ids=COLLISION_IDS)
def test_matrix_id_rejects_existing_canonical_id_column(
    source_id,
    schema,
    data,
    helpers,
):

    df = helpers.create_df(data, schema)

    with pytest.raises(ValueError, match=re.escape(INVALID_CANONICAL_ID_COLLISION_MSG.format("canonical_id"))):
        lk.dedupe(df).apply(lk.exact()).canonicalize(SINGLE_COL, source_id=source_id)
