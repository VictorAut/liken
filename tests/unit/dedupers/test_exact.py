"""Liken deduper unit tests"""

from unittest.mock import Mock

import pyarrow as pa
import pytest

import liken as lk


#################################
# exact bucket keys             #
#################################


def test_exact_nulls_pair_with_nulls_only(mock_df):
    """A null groups only with another null; literal "na" is an ordinary value."""
    mock_df.get_array = Mock(return_value=pa.array(["na", None, "na", None]))

    uf, n = lk.exact().set_frame(mock_df).build_union_find("address", [])

    assert n == 4
    assert uf[0] == uf[2]  # the two "na" literals group as values
    assert uf[1] == uf[3]  # the two nulls group
    assert uf[0] != uf[1]  # a literal never groups with a null


@pytest.mark.parametrize("values", [[7, None, 7, None], [1.5, None, 1.5, None]], ids=["int", "float"])
def test_exact_non_string_columns_group_equal_values(mock_df, values):
    """Non-string columns group equal values; nulls group with nulls."""
    mock_df.get_array = Mock(return_value=pa.array(values))

    uf, n = lk.exact().set_frame(mock_df).build_union_find("address", [])

    assert n == 4
    assert uf[0] == uf[2]
    assert uf[1] == uf[3]
    assert uf[0] != uf[1]


def test_exact_nan_groups_with_null(mock_df):
    """NaN is missing on the same footing as None: one missing group."""
    mock_df.get_array = Mock(return_value=pa.array([1.0, None, float("nan"), None]))

    uf, n = lk.exact().set_frame(mock_df).build_union_find("address", [])

    assert n == 4
    assert uf[1] == uf[2] == uf[3]  # the two nulls and the NaN form one group
    assert uf[0] != uf[1]


def test_exact_nested_typed_column_groups_by_content(mock_df):
    """Nested-typed columns keep grouping by content.

    Bucket keys stay pa.Scalar, which is hashable for list/struct types;
    normalising through as_py() would raise TypeError on the unhashable
    Python list or dict a nested scalar converts to.
    """
    mock_df.get_array = Mock(return_value=pa.array([[1], [1], [2]], type=pa.list_(pa.int64())))

    uf, n = lk.exact().set_frame(mock_df).build_union_find("address", [])

    assert n == 3
    assert uf[0] == uf[1]  # equal contents group
    assert uf[0] != uf[2]


def test_exact_compound_missing_members_collapse(mock_df):
    """Missing members collapse per member: (None, "x") and (NaN, "x") are one key."""
    mock_df.get_array = Mock(
        return_value=pa.table({"a": pa.array([None, float("nan"), None]), "b": pa.array(["x", "x", "y"])})
    )

    uf, n = lk.exact().set_frame(mock_df).build_union_find(("a", "b"), [])

    assert n == 3
    assert uf[0] == uf[1]  # (None, "x") groups with (NaN, "x")
    assert uf[0] != uf[2]  # (None, "x") does not group with (None, "y")
