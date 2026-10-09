"""Narrow integration tests for true null handling through the public API.

Each case pins one facet of the null contract (the D1-D8 defects): a missing
value — None, and float NaN where the backend keeps it — groups only with
another missing value, is never scored, and never satisfies a predicate.
"""

from __future__ import annotations

import pytest

import liken as lk
from liken.constants import CANONICAL_ID
from liken.core.missing import is_missing


# A custom deduper that asserts the raw-values contract from the inside: it
# is invoked with the column's values and fails unless missing values arrive
# unsubstituted. Assertions rather than recorded state because distributed
# backends (dask, ray, pyspark) pickle the callable to workers, where module
# globals are copies. The fixture holds one partition per backend, so every
# invocation sees the missing row; a repartitioned frame would need a
# weaker assertion.
@lk.custom.register
def require_raw_values(array):
    missing = [value for value in array if is_missing(value)]

    assert missing  # a missing value arrived, not a placeholder string
    assert "na" not in array

    yield from ()


# fmt: off

CASES = [
    # D1: a literal "na" is an ordinary value; nulls group with nulls only
    (
        "exact-null-vs-literal-na",
        ["address"],
        [("na",), (None,), ("na",), (None,)],
        lk.col("address").exact(),
        [0, 1, 0, 1],
    ),
    # D1/D8: null does not group with "" or literal "na"; the values pair
    # among themselves ("na" and "" are 2 edits apart)
    (
        "edit-distance-null-vs-values",
        ["address"],
        [("na",), (None,), ("na",), ("",)],
        lk.col("address").edit_distance(max_distance=2),
        [0, 1, 0, 0],
    ),
    # D2: a null never satisfies a positive predicate; the literal "na" does
    (
        "str-startswith-null-never-matches",
        ["address"],
        [("nathan",), (None,), ("na",)],
        lk.col("address").str_startswith(pattern="n"),
        [0, 1, 0],
    ),
    # D2: str_len cannot measure a null, so a null never matches
    (
        "str-len-null-never-matches",
        ["address"],
        [("ab",), (None,)],
        lk.col("address").str_len(min_len=1, max_len=3),
        [0, 1],
    ),
    # D3: a null never satisfies a negated predicate, whether or not None is
    # in the membership values
    (
        "negated-isin-nulls-never-match",
        ["address"],
        [(None,), (None,), ("x",), ("x",)],
        ~lk.col("address").isin(values=["x"]),
        [0, 1, 2, 3],
    ),
    (
        "negated-isin-null-listed-still-never-matches",
        ["address"],
        [(None,), (None,), ("x",)],
        ~lk.col("address").isin(values=["x", None]),
        [0, 1, 2],
    ),
    # D3: negated str_len also leaves nulls unmatched
    (
        "negated-str-len-null-never-matches",
        ["address"],
        [("abcd",), ("abcde",), (None,)],
        ~lk.col("address").str_len(min_len=1, max_len=2),
        [0, 0, 2],
    ),
    # D4: at any threshold a null never pairs with a value; the values pair
    # among themselves (ratio("nathan", "nadia") is ~54.5)
    (
        "fuzzy-nulls-never-pair-with-values",
        ["address"],
        [("nathan",), (None,), ("nadia",), (None,)],
        lk.col("address").fuzzy(threshold=0.5),
        [0, 1, 0, 1],
    ),
    # D4: nulls group with nulls at the default ngram=3, where they did not
    # before ("nathan" and "nadia" share no character 3-grams)
    (
        "tfidf-nulls-group-at-default-ngram",
        ["address"],
        [("nathan",), (None,), ("nadia",), (None,)],
        lk.col("address").tfidf(),
        [0, 1, 2, 1],
    ),
    (
        "lsh-nulls-group-at-default-ngram",
        ["address"],
        [("nathan",), (None,), ("nadia",), (None,)],
        lk.col("address").lsh(),
        [0, 1, 2, 1],
    ),
    # D5: non-string columns deduplicate without the placeholder coalesce
    (
        "exact-int-column",
        ["number"],
        [(7,), (None,), (7,), (None,)],
        lk.col("number").exact(),
        [0, 1, 0, 1],
    ),
    (
        "exact-float-column",
        ["number"],
        [(1.5,), (None,), (1.5,), (None,)],
        lk.col("number").exact(),
        [0, 1, 0, 1],
    ),
    # D6: a column of all nulls forms one group, no engine input, no crash
    (
        "all-null-exact",
        ["address"],
        [(None,), (None,), (None,)],
        lk.col("address").exact(),
        [0, 0, 0],
    ),
    (
        "all-null-fuzzy",
        ["address"],
        [(None,), (None,), (None,)],
        lk.col("address").fuzzy(),
        [0, 0, 0],
    ),
    (
        "all-null-edit-distance",
        ["address"],
        [(None,), (None,)],
        lk.col("address").edit_distance(),
        [0, 0],
    ),
    (
        "all-null-tfidf",
        ["address"],
        [(None,), (None,), (None,)],
        lk.col("address").tfidf(),
        [0, 0, 0],
    ),
    (
        "all-null-lsh",
        ["address"],
        [(None,), (None,), (None,)],
        lk.col("address").lsh(),
        [0, 0, 0],
    ),
    # D7: single and multi-column paths agree - compound keys collapse
    # missing members, so (None, "x") and (NaN, "x") are one key. Backends
    # that convert NaN to null on the way into Arrow (pandas, modin, dask,
    # ray) deliver None here; polars and pyarrow keep float NaN; both are
    # one missing class and group identically.
    (
        "compound-missing-members-collapse",
        ["a", "b"],
        [(None, "x"), (float("nan"), "x"), (None, "y")],
        lk.col(("a", "b")).exact(),
        [0, 0, 2],
    ),
    # predicate polarities: isna matches None and NaN; ~isna groups only the
    # non-missing values; positive isin matches a null iff None is listed
    (
        "isna-matches-none-and-nan",
        ["number"],
        [(None,), (float("nan"),), (1.0,)],
        lk.col("number").isna(),
        [0, 0, 2],
    ),
    (
        "notna-matches-only-non-missing",
        ["number"],
        [(None,), (float("nan"),), (1.0,), (2.0,)],
        ~lk.col("number").isna(),
        [0, 1, 2, 2],
    ),
    (
        "isin-null-matches-iff-none-listed",
        ["address"],
        [(None,), (None,), ("x",)],
        lk.col("address").isin(values=["x", None]),
        [0, 0, 0],
    ),
    (
        "isin-null-unlisted-never-matches",
        ["address"],
        [(None,), (None,), ("x",)],
        lk.col("address").isin(values=["x"]),
        [0, 1, 2],
    ),
]

# fmt: on


@pytest.mark.parametrize(
    "schema, data, step, expected_canonical_id",
    [case[1:] for case in CASES],
    ids=[case[0] for case in CASES],
)
def test_matrix_nulls(schema, data, step, expected_canonical_id, helpers, spark_session, request):

    if request.config.getoption("--backend") == "pyspark" and request.node.callspec.id.startswith("all-null"):
        pytest.skip("spark cannot infer column types from an all-null dataset")

    df = helpers.create_df(data, schema)

    df = lk.dedupe(df, spark_session=spark_session).apply(lk.pipeline().step(step)).canonicalize().collect()

    assert helpers.get_column_as_list(df, CANONICAL_ID) == expected_canonical_id


@pytest.mark.parametrize(
    "schema, data",
    [
        (["address"], [("a",), (None,), ("b",)]),
        # the float case: backends that convert NaN to null on the way into
        # Arrow deliver None; polars and pyarrow keep float NaN; both are
        # raw missing values for a custom deduper
        (["number"], [(1.5,), (float("nan"),), (2.5,)]),
    ],
    ids=["string-column", "float-column"],
)
def test_custom_deduper_receives_raw_values(schema, data, helpers, spark_session):
    """A registered custom deduper sees raw values, nulls included."""
    df = helpers.create_df(data, schema)

    out = (
        lk.dedupe(df, spark_session=spark_session)
        .apply(lk.pipeline().step(lk.col(schema[0]).require_raw_values()))
        .canonicalize()
        .collect()
    )

    # reading the column forces the compute, so the callable's assertions
    # also run on the lazy backends (dask, pyspark)
    column = helpers.get_column_as_list(out, schema[0])

    assert column[0] == data[0][0]
    assert is_missing(column[1])
    assert column[2] == data[2][0]


def test_empty_frame_canonicalise_and_drop(helpers, spark_session, request):
    """Empty frames pass through canonicalize and drop_duplicates (D6)."""
    backend = request.config.getoption("--backend")

    if backend == "ray":
        pytest.skip(
            "ray reports an unknown schema for empty transformed datasets, so the "
            "canonical-id check cannot run; empty-frame passthrough is unsupported on ray"
        )

    if backend == "pyspark":
        pytest.skip("spark cannot infer a schema from an empty dataset")

    df = helpers.create_df([], ["address"])

    out = lk.dedupe(df, spark_session=spark_session).apply(lk.exact()).canonicalize("address", id="address").collect()

    assert helpers.get_column_as_list(out, "address") == []

    dropped = lk.dedupe(df, spark_session=spark_session).apply(lk.exact()).drop_duplicates("address", keep="first")

    assert helpers.get_column_as_list(dropped, "address") == []
