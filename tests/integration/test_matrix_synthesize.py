"""Narrow integration tests for creation of canonical ID column"""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

import liken as lk
from liken.constants import CANONICAL_ID


# SET UP:

SCHEMA = ["uid", "address"]
SINGLE_COL = "address"


PARAMS = [
    (
        ["address", "email", "state", "random", "country"],
        [
            ["123ab, OL5", "hello@example.com", None, "A", None],
            ["123ab, OL5", None, None, "B", "Germany"],
            ["Completely random address", None, None, "C", "Spain"],
            ["55 bling blong road", None, None, "D", "Ireland"],
            ["55 bling blong road", "byebye@aol.ir", "Texas", "E", None],
        ],
    ),
]


@pytest.mark.parametrize("schema, data", PARAMS)
@pytest.mark.parametrize("backend", ["pandas", "polars", "spark"])
def test_matrix_preprocessors(
    backend,
    schema,
    data,
    spark_session,
    helpers,
    request
):

    backend = request.config.getoption("--backend")

    df = helpers.create_df(backend, spark_session, data, schema)

    result = lk.dedupe(df, spark_session=spark_session).apply(lk.exact()).canonicalize("address")

    df = result.collect()
    synthesized = result.synthesize()
    canonicals = result.canonicals()

    # base line check, retuns 5 rows.
    assert helpers.get_column_as_list(df, CANONICAL_ID) == [0, 0, 2, 3, 3]

    # returns 3 rows!
    assert helpers.get_column_as_list(synthesized, "address") == [
        "123ab, OL5",
        "Completely random address",
        "55 bling blong road",
    ]
    assert helpers.get_column_as_list(synthesized, "email") == [
        "hello@example.com",
        None,
        "byebye@aol.ir",
    ]
    assert helpers.get_column_as_list(synthesized, "state") == [
        None,
        None,
        "Texas",
    ]
    assert helpers.get_column_as_list(synthesized, "random") == [
        "A",
        "C",
        "D",
    ]
    assert helpers.get_column_as_list(synthesized, "country") == [
        "Germany",
        "Spain",
        "Ireland",
    ]

    # And assert canonicals
    assert canonicals == {0: 2, 3: 2}
