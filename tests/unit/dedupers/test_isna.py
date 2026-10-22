"""Liken deduper unit tests"""

from unittest.mock import Mock

import pyarrow as pa

import liken as lk


##################################
# isna NaN and None semantics #
#################################


def test_isna_groups_none_and_nan_together(mock_df):
    mock_df.get_array = Mock(return_value=pa.array([None, float("nan"), 1.0]))

    uf, n = lk.isna().set_frame(mock_df).build_union_find("address", [])

    assert n == 3
    assert uf[0] == uf[1]  # None and NaN group together
    assert uf[0] != uf[2]


def test_notna_groups_only_non_null_values(mock_df):
    mock_df.get_array = Mock(return_value=pa.array([None, float("nan"), 1.0, 2.0]))

    uf, n = (~lk.isna()).set_frame(mock_df).build_union_find("address", [])

    assert n == 4
    assert uf[2] == uf[3]  # only the non-null values group
    assert uf[0] != uf[2] and uf[1] != uf[2]
