"""Pandas Nulls, especially with the Pandas PyArrow backend, i.e. v2 and up
often have hard to understand Null Types"""

from __future__ import annotations

import pandas as pd
import pytest

import liken as lk
from liken.constants import CANONICAL_ID


# CONSTANTS:


SINGLE_COL = "address"


PARAMS = [
    (
        [
            [1, None],
            [2, None],
            [3, "random"],
        ]
    ),
    (
        [
            [1, None],
            [2, pd.NA],
            [3, "random"],
        ]
    ),
    (
        [
            [1, pd.NA],
            [2, pd.NA],
            [3, "random"],
        ]
    ),
]
IDS = [
    "None-None",
    "None-pd.NA",
    "pd.NA-pd.NA",
]


@pytest.mark.parametrize("data", PARAMS, ids=IDS)
@pytest.mark.parametrize(
    "deduper, expected_ids",
    [
        (lk.exact(), [1, 1, 3]),
        (lk.isna(), [1, 1, 3]),
        (~lk.isna(), [1, 2, 3]),
    ],
    ids=["exact", "isna", "notna"],
)
def test_matrix_pd_nulls(data, deduper, expected_ids, helpers, request):

    backend = request.config.getoption("--backend")

    if backend != "pandas":
        pytest.skip("Pandas only test")

    df = pd.DataFrame(columns=["id", "address"], data=data)

    df_deduped = lk.dedupe(df).apply(deduper).canonicalize("address", id="id").collect()

    assert helpers.get_column_as_list(df_deduped, CANONICAL_ID) == expected_ids