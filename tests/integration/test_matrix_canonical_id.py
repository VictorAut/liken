"""Narrow integration tests for the `canonical_id` output-label parameter.

The label threads through canonicalize -> canonicals -> synthesize, and a
second enact on a custom-labelled frame leaves the column in place.
"""

from __future__ import annotations

import liken as lk
from liken.constants import CANONICAL_ID


CUSTOM_LABEL = "master_id"

DATA = [
    ["1 High St", "a@x.com"],
    ["1 High St", "b@x.com"],
    ["2 Low St", "c@x.com"],
]
SCHEMA = ["address", "email"]


def column_labels(df):
    import pyarrow as pa

    if isinstance(df, pa.Table):
        return [field.name for field in df.schema]
    backend_columns = getattr(df, "columns", None)
    return list(backend_columns() if callable(backend_columns) else backend_columns)


def test_canonicalize_custom_label(helpers, spark_session):
    """`canonical_id=master_id` produces a `master_id` column."""
    df = helpers.create_df(DATA, SCHEMA)

    result = (
        lk.dedupe(df, spark_session=spark_session)
        .apply(lk.exact())
        .canonicalize("address", canonical_id=CUSTOM_LABEL)
        .collect()
    )

    # exact ids prove the label survived any partition round-trip
    assert helpers.get_column_as_list(result, CUSTOM_LABEL) == [0, 0, 2]
    assert CUSTOM_LABEL in column_labels(result)
    assert CANONICAL_ID not in column_labels(result)


def test_canonicals_and_synthesize_honour_custom_label(helpers, spark_session):
    """Post-hoc reads use the stored label."""
    df = helpers.create_df(DATA, SCHEMA)

    dupe = (
        lk.dedupe(df, spark_session=spark_session).apply(lk.exact()).canonicalize("address", canonical_id=CUSTOM_LABEL)
    )

    assert dupe.canonicals() == {0: 2}

    synthesized = dupe.synthesize()

    assert helpers.get_column_as_list(synthesized, CUSTOM_LABEL) == [0, 2]
    assert helpers.get_column_as_list(synthesized, "email") == ["a@x.com", "c@x.com"]


def test_drop_duplicates_after_custom_label_canonicalize(helpers, spark_session):
    """A second enact on a custom-labelled frame leaves the column in place."""
    df = helpers.create_df(DATA, SCHEMA)

    dupe = (
        lk.dedupe(df, spark_session=spark_session)
        .apply(lk.exact())
        .canonicalize("address", canonical_id=CUSTOM_LABEL)
        .collect()
    )

    result = lk.dedupe(dupe, spark_session=spark_session).apply(lk.exact()).drop_duplicates("email")

    assert helpers.get_column_as_list(result, "email") == ["a@x.com", "b@x.com", "c@x.com"]
    assert CUSTOM_LABEL in column_labels(result)


def test_drop_duplicates_keeps_data_column_named_canonical_id(helpers, spark_session):
    """A data column labelled like the canonical id is data, never consumed."""
    df = helpers.create_df([[101, "a@x"], [102, "a@x"], [103, "b@x"]], ["canonical_id", "email"])

    result = lk.dedupe(df, spark_session=spark_session).drop_duplicates("email")

    assert "canonical_id" in column_labels(result)
    assert helpers.get_column_as_list(result, "canonical_id") == [101, 103]
    assert helpers.get_column_as_list(result, "email") == ["a@x", "b@x"]
