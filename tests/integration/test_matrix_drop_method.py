"""Integration tests for the public `.drop_duplicates()` method.

Distinct from `test_matrix_pd_affordances.py`, which is pandas-only by
design (pandas affordances registered in the top-level `__init__`). This
matrix test exercises the public method itself across all backends.
"""

from __future__ import annotations

import pytest

import liken as lk
from liken.constants import CANONICAL_ID


# fake_10 address groups: rows 1 & 8 share an address, rows 5 & 9 are null
# and nulls are grouped via the NA placeholder, so two rows drop in total
EXPECTED_IDS_FIRST = [1, 2, 3, 4, 5, 6, 7, 10]
EXPECTED_IDS_LAST = [2, 3, 4, 6, 7, 8, 9, 10]


def column_labels(df):
    backend_columns = getattr(df, "columns", None)
    return list(backend_columns() if callable(backend_columns) else backend_columns)


@pytest.mark.parametrize("keep, expected_ids", [("first", EXPECTED_IDS_FIRST), ("last", EXPECTED_IDS_LAST)])
def test_matrix_drop_method(keep, expected_ids, dataframe, helpers, spark_session):

    df = lk.dedupe(dataframe, spark_session=spark_session).apply(lk.exact()).drop_duplicates("address", keep=keep)

    assert CANONICAL_ID not in column_labels(df)
    assert sorted(helpers.get_column_as_list(df, "id")) == expected_ids


def test_drop_method_without_apply_is_exact(dataframe, helpers, spark_session, request):

    backend = request.config.getoption("--backend")
    if backend not in ("pandas", "polars"):
        pytest.skip("Implicit-exact test run on pandas and polars only")

    # implicit exact: no .apply() before .drop_duplicates()
    implicit = lk.dedupe(dataframe, spark_session=spark_session).drop_duplicates("address", keep="first")

    explicit = (
        lk.dedupe(dataframe, spark_session=spark_session).apply(lk.exact()).drop_duplicates("address", keep="first")
    )

    implicit_ids = helpers.get_column_as_list(implicit, "id")
    explicit_ids = helpers.get_column_as_list(explicit, "id")

    assert sorted(implicit_ids) == sorted(explicit_ids) == EXPECTED_IDS_FIRST
