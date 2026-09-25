"""Integration tests for the Spark Rows path.

`PysparkRows` is the wrapper used per partition on Spark worker nodes, fed
by `Dedupe._from_rows`. It runs in the driver process here, so it is also
measured by coverage.
"""

from __future__ import annotations

import pytest

import liken as lk
from liken.constants import CANONICAL_ID
from liken.core.dispatcher import get_backend
from liken.liken import Dedupe


# HELPERS:


def make_rows():
    from pyspark.sql import Row

    return [
        Row(id=1, address="london"),
        Row(id=2, address="london"),
        Row(id=3, address="paris"),
    ]


@pytest.fixture(autouse=True)
def pyspark_only(request):
    backend = request.config.getoption("--backend")
    if backend != "pyspark":
        pytest.skip("Pyspark only test")


# DISPATCH:


def test_backend_dispatches_list_of_rows():
    backend = get_backend(make_rows())

    assert backend.name == "pyspark"


def test_wrap_returns_spark_rows_wrapper():
    from liken.backends.pyspark.wrapper import PysparkRows

    assert isinstance(get_backend(make_rows()).wrap(make_rows()), PysparkRows)


# WRAPPER METHODS:


def test_get_col():
    wdf = get_backend(make_rows()).wrap(make_rows())

    arr = wdf._get_col("address")

    assert arr.to_pylist() == ["london", "london", "paris"]


def test_get_cols():
    wdf = get_backend(make_rows()).wrap(make_rows())

    table = wdf._get_cols(("id", "address"))

    assert table.column("id").to_pylist() == [1, 2, 3]
    assert table.column("address").to_pylist() == ["london", "london", "paris"]


def test_put_col():
    wdf = get_backend(make_rows()).wrap(make_rows())

    wdf.put_col(CANONICAL_ID, [0, 0, 1])

    assert wdf._get_col(CANONICAL_ID).to_pylist() == [0, 0, 1]


def test_drop_duplicates():
    # note: drop_duplicates mutates in place and returns self, so the two
    # keeps need separate wrappers
    wdf_first = get_backend(make_rows()).wrap(make_rows())
    wdf_first.put_col(CANONICAL_ID, [0, 0, 1])
    assert wdf_first.drop_duplicates(keep="first")._get_col("id").to_pylist() == [1, 3]

    wdf_last = get_backend(make_rows()).wrap(make_rows())
    wdf_last.put_col(CANONICAL_ID, [0, 0, 1])
    assert wdf_last.drop_duplicates(keep="last")._get_col("id").to_pylist() == [2, 3]


# INTERNAL CONSTRUCTOR:


def test_from_rows_canonicalizes():
    from pyspark.sql import Row

    # mimic the worker-node input: rows carry canonical_id, added upstream
    # by PysparkDF when the RDD is created
    rows = [
        Row(id=1, address="london", canonical_id=0),
        Row(id=2, address="london", canonical_id=0),
        Row(id=3, address="paris", canonical_id=1),
    ]

    result = Dedupe._from_rows(rows).apply(lk.exact()).canonicalize("address", drop_duplicates=True).collect()

    ids = [row["id"] for row in result]
    assert ids == [1, 3]
