"""Integration tests for the `canonical_id` output-label parameter.

The label threads through canonicalize -> canonicals -> synthesize and, on
partitioned backends, across the partition boundary including the drop path.
"""

from __future__ import annotations

import re

import pytest

import liken as lk
from liken.constants import CANONICAL_ID
from liken.constants import INVALID_CANONICAL_ID_COLLISION_MSG


CUSTOM_LABEL = "master_id"

DATA = [
    ["1 High St", "a@x.com"],
    ["1 High St", "b@x.com"],
    ["2 Low St", "c@x.com"],
]
SCHEMA = ["address", "email"]


def test_canonicalize_custom_label(helpers, dataframe, spark_session):
    """`canonical_id=master_id` produces a `master_id` column."""
    df = helpers.create_df(DATA, SCHEMA)

    result = (
        lk.dedupe(df, spark_session=spark_session)
        .apply(lk.exact())
        .canonicalize("address", canonical_id=CUSTOM_LABEL)
        .collect()
    )

    assert helpers.get_column_as_list(result, CUSTOM_LABEL) == [0, 0, 2]
    assert not _has_column(result, CANONICAL_ID, helpers)


def test_canonicals_and_synthesize_honour_custom_label(helpers, dataframe, spark_session):
    """Probe-C regression: post-hoc reads use the stored label."""
    df = helpers.create_df(DATA, SCHEMA)

    dupe = (
        lk.dedupe(df, spark_session=spark_session).apply(lk.exact()).canonicalize("address", canonical_id=CUSTOM_LABEL)
    )

    assert dupe.canonicals() == {0: 2}

    synthesized = dupe.synthesize()

    assert helpers.get_column_as_list(synthesized, CUSTOM_LABEL) == [0, 2]
    assert helpers.get_column_as_list(synthesized, "email") == ["a@x.com", "c@x.com"]


def test_canonicalize_rejects_existing_column(helpers, dataframe, spark_session):
    """A `canonical_id` colliding with an existing column raises before execution."""
    df = helpers.create_df(DATA, SCHEMA)

    dupe = lk.dedupe(df, spark_session=spark_session).apply(lk.exact())

    with pytest.raises(ValueError, match=re.escape(INVALID_CANONICAL_ID_COLLISION_MSG.format("email"))):
        dupe.canonicalize("address", canonical_id="email")


def test_source_id_copies_into_custom_label(helpers, dataframe, spark_session):
    """source-name != output-name: `uid` values land in `master_id`."""
    df = helpers.create_df(
        [
            ["u1", "1 High St", "a@x.com"],
            ["u2", "1 High St", "b@x.com"],
            ["u3", "2 Low St", "c@x.com"],
        ],
        ["uid", "address", "email"],
    )

    result = (
        lk.dedupe(df, spark_session=spark_session)
        .apply(lk.exact())
        .canonicalize("address", source_id="uid", canonical_id=CUSTOM_LABEL)
        .collect()
    )

    assert helpers.get_column_as_list(result, CUSTOM_LABEL) == ["u1", "u1", "u3"]


def test_drop_duplicates_after_custom_label_canonicalize(helpers, dataframe, spark_session):
    """A second enact on a custom-labelled frame works: the default drop path
    is untouched by the stored label (probe-B regression)."""
    df = helpers.create_df(DATA, SCHEMA)

    dupe = (
        lk.dedupe(df, spark_session=spark_session)
        .apply(lk.exact())
        .canonicalize("address", canonical_id=CUSTOM_LABEL)
        .collect()
    )

    result = lk.dedupe(dupe, spark_session=spark_session).apply(lk.exact()).drop_duplicates("email")

    assert helpers.get_column_as_list(result, "email") == ["a@x.com", "b@x.com", "c@x.com"]


# HELPERS:


def _has_column(df, label: str, helpers) -> bool:
    try:
        helpers.get_column_as_list(df, label)
    except (KeyError, Exception):
        return False
    return True


# PARTITIONED BACKENDS:


def test_partitioned_custom_label_end_to_end(helpers, request, spark_session):
    """Dask, Ray and PySpark thread the custom label across partitions,
    including the drop path."""
    backend = request.config.getoption("--backend")

    if backend not in ("dask", "ray", "pyspark"):
        pytest.skip("Partitioned backends only")

    df = helpers.create_df(DATA, SCHEMA)

    dupe = (
        lk.dedupe(df, spark_session=spark_session).apply(lk.exact()).canonicalize("address", canonical_id=CUSTOM_LABEL)
    )

    result = dupe.collect()

    labels = helpers.get_column_as_list(result, CUSTOM_LABEL)
    assert len(labels) == 3
    assert len(set(labels)) < len(labels)  # duplicates exist, so they were grouped
    assert not _has_column(result, CANONICAL_ID, helpers)

    # post-hoc reads honour the label collected from the workers
    assert dupe.canonicals() == {0: 2}

    # the drop path removes the custom-labelled column
    dropped = dupe.drop_duplicates("email")

    assert not _has_column(dropped, CUSTOM_LABEL, helpers)
    # drop dedups on the stored canonical label, same as the default label path
    assert helpers.get_column_as_list(dropped, "email") == ["a@x.com", "c@x.com"]
