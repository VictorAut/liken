"""Pandas Nulls, especially with the Pandas PyArrow backend, i.e. v2 and up
often have hard to understand Null Types"""

from __future__ import annotations

import pandas as pd
import pytest

from liken import Dedupe
from liken import exact
from liken._constants import CANONICAL_ID
from liken.rules import isna


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
def test_matrix_exact_on_na(data, helpers):

    df = pd.DataFrame(columns=["id", "address"], data=data)

    lk = Dedupe(df)
    lk.apply(exact())
    df_deduped = lk.canonicalize("address", id="id")

    assert helpers.get_column_as_list(df_deduped, CANONICAL_ID) == [1, 1, 3]


@pytest.mark.parametrize("data", PARAMS, ids=IDS)
def test_matrix_isna_on_na(data, helpers):

    df = pd.DataFrame(columns=["id", "address"], data=data)

    lk = Dedupe(df)
    lk.apply(isna())
    df_deduped = lk.canonicalize("address", id="id")

    assert helpers.get_column_as_list(df_deduped, CANONICAL_ID) == [1, 1, 3]


@pytest.mark.parametrize("data", PARAMS, ids=IDS)
def test_matrix_notna_on_na(data, helpers):

    df = pd.DataFrame(columns=["id", "address"], data=data)

    lk = Dedupe(df)
    lk.apply(~isna())
    df_deduped = lk.canonicalize("address", id="id")

    assert helpers.get_column_as_list(df_deduped, CANONICAL_ID) == [1, 2, 3]
