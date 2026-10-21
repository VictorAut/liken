"""Narrow integration tests for the date_range predicate deduper.

The fake_10 fixture has no date column, so the frames are built inline
(helpers.create_df) like test_matrix_nulls.py, and run on every backend
through the pipeline API.
"""

from __future__ import annotations

import pytest

import liken as lk
from liken.constants import CANONICAL_ID
from liken.exceptions import InvalidArgumentValueError


# fmt: off

CASES = [
    (
        # window=0 is same-date matching; the different dates stay apart
        "same-date-window-zero",
        ["dob"],
        [("2024-01-05",), ("2024-01-05",), ("2024-06-01",)],
        lk.col("dob").date_range(format="%Y-%m-%d"),
        [0, 0, 2],
    ),
    (
        # Jan 5 and Jan 7 are 2 days apart: the inclusive bound merges them
        "window-two-inclusive-boundary",
        ["dob"],
        [("2024-01-05",), ("2024-01-07",), ("2024-01-10",)],
        lk.col("dob").date_range(format="%Y-%m-%d", window=2),
        [0, 0, 2],
    ),
    (
        # adjacency at window=1: pairs 5-6 and 6-7 match, and union-find
        # merges the chain transitively (5 and 7 are 2 days apart)
        "window-one-adjacent-chain",
        ["dob"],
        [("2024-01-05",), ("2024-01-06",), ("2024-01-07",)],
        lk.col("dob").date_range(format="%Y-%m-%d", window=1),
        [0, 0, 0],
    ),
    (
        # nulls group together; a null never merges with a date
        "nulls-group-together",
        ["dob"],
        [("2024-01-05",), (None,), (None,), ("2024-06-01",)],
        lk.col("dob").date_range(format="%Y-%m-%d", window=30),
        [0, 1, 1, 3],
    ),
]

NEGATION_CASES = [
    (
        # ~date_range groups the values that fail to parse; parsed values
        # and nulls stay apart
        "negated-groups-unparseable",
        ["dob"],
        [("garbage",), ("also garbage",), ("2024-01-05",), (None,)],
        ~lk.col("dob").date_range(format="%Y-%m-%d"),
        [0, 0, 2, 3],
    ),
]

# fmt: on


@pytest.mark.parametrize(
    "schema, data, step, expected_canonical_id",
    [case[1:] for case in CASES + NEGATION_CASES],
    ids=[case[0] for case in CASES + NEGATION_CASES],
)
def test_matrix_date_range(schema, data, step, expected_canonical_id, helpers):

    df = helpers.create_df(data, schema)

    df = lk.dedupe(df).apply(lk.pipeline().step(step)).canonicalize().collect()

    assert helpers.get_column_as_list(df, CANONICAL_ID) == expected_canonical_id


def test_matrix_date_range_parse_failure_raises(helpers):

    df = helpers.create_df([("2024-01-05",), ("05/01/2024",)], ["dob"])

    with pytest.raises(InvalidArgumentValueError, match="05/01/2024"):
        lk.dedupe(df).apply(lk.pipeline().step(lk.col("dob").date_range(format="%Y-%m-%d"))).canonicalize().collect()
