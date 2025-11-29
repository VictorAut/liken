from __future__ import annotations
from unittest.mock import Mock, patch, call

import numpy as np
import pytest

from dupegrouper.base import _wrap
from dupegrouper.definitions import TMP_ATTR_LABEL, GROUP_ID
from dupegrouper.strategies.strdedupers import (
    StrStartsWith,
    StrEndsWith,
    StrContains,
)


####################################
# STR STARTS WITH DEDUPE UNIT TEST #
####################################


def test_str_starts_dedupe_unit():
    attr = "address"
    dummy_array = np.array(["foo", "bar", "bar"])

    deduper = StrStartsWith(pattern="b")

    mock_wrapped_df = Mock()
    mock_wrapped_df.get_col.return_value = dummy_array
    deduper.wrapped_df = mock_wrapped_df

    with patch.object(
        deduper,
        "assign_group_id",
        return_value=mock_wrapped_df,
    ) as mock_assign_group_id:

        # Also mock wrapped_df chaining methods
        mock_wrapped_df.map_dict.return_value = [None, "bar", "bar"]
        mock_wrapped_df.put_col.return_value = mock_wrapped_df
        mock_wrapped_df.assign_group_id.return_value = mock_wrapped_df
        mock_wrapped_df.drop_col.return_value = mock_wrapped_df

        # Run dedupe
        result = deduper.dedupe(attr)

        mock_wrapped_df.map_dict.assert_called_once_with(attr, {"bar": "bar"})

        # second put call is part of assign_group_id which in another unit test
        put_col_call = mock_wrapped_df.put_col.call_args_list[0]
        assert put_col_call == call(TMP_ATTR_LABEL, [None, "bar", "bar"])

        mock_assign_group_id.assert_called_once()
        mock_wrapped_df.drop_col.assert_called_once()

        assert result == mock_wrapped_df


def test_str_ends_dedupe_unit():
    attr = "address"
    dummy_array = np.array(["foo", "bar", "bar"])

    deduper = StrEndsWith(pattern="ar")

    mock_wrapped_df = Mock()
    mock_wrapped_df.get_col.return_value = dummy_array
    deduper.wrapped_df = mock_wrapped_df

    with patch.object(
        deduper,
        "assign_group_id",
        return_value=mock_wrapped_df,
    ) as mock_assign_group_id:

        # Also mock wrapped_df chaining methods
        mock_wrapped_df.map_dict.return_value = [None, "bar", "bar"]
        mock_wrapped_df.put_col.return_value = mock_wrapped_df
        mock_wrapped_df.assign_group_id.return_value = mock_wrapped_df
        mock_wrapped_df.drop_col.return_value = mock_wrapped_df

        # Run dedupe
        result = deduper.dedupe(attr)

        mock_wrapped_df.map_dict.assert_called_once_with(attr, {"bar": "bar"})

        # second put call is part of assign_group_id which in another unit test
        put_col_call = mock_wrapped_df.put_col.call_args_list[0]
        assert put_col_call == call(TMP_ATTR_LABEL, [None, "bar", "bar"])

        mock_assign_group_id.assert_called_once()
        mock_wrapped_df.drop_col.assert_called_once()

        assert result == mock_wrapped_df


def test_str_contains_dedupe_unit():
    attr = "address"
    dummy_array = np.array(["foo", "bar", "bar"])

    deduper = StrContains(pattern="a")

    mock_wrapped_df = Mock()
    mock_wrapped_df.get_col.return_value = dummy_array
    deduper.wrapped_df = mock_wrapped_df

    with patch.object(
        deduper,
        "assign_group_id",
        return_value=mock_wrapped_df,
    ) as mock_assign_group_id:

        # Also mock wrapped_df chaining methods
        mock_wrapped_df.map_dict.return_value = [None, "bar", "bar"]
        mock_wrapped_df.put_col.return_value = mock_wrapped_df
        mock_wrapped_df.assign_group_id.return_value = mock_wrapped_df
        mock_wrapped_df.drop_col.return_value = mock_wrapped_df

        # Run dedupe
        result = deduper.dedupe(attr)

        mock_wrapped_df.map_dict.assert_called_once_with(attr, {"bar": "bar"})

        # second put call is part of assign_group_id which in another unit test
        put_col_call = mock_wrapped_df.put_col.call_args_list[0]
        assert put_col_call == call(TMP_ATTR_LABEL, [None, "bar", "bar"])

        mock_assign_group_id.assert_called_once()
        mock_wrapped_df.drop_col.assert_called_once()

        assert result == mock_wrapped_df


##################################
# DEDUPE NARROW INTEGRATION TEST #
##################################

star_starts_parametrize_data = [
    # i.e. no deduping because no string starts with the pattern
    ({"pattern": "zzzzz", "case": True}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    ({"pattern": "zzzzz", "case": False}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # String does dedupe if case correct; but doesn't otherwise
    ({"pattern": "calle", "case": True}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    ({"pattern": "calle", "case": False}, [1, 2, 3, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
]


@pytest.mark.parametrize("input, output", star_starts_parametrize_data)
def test_str_starts_dedupe_integrated(input, output, dataframe, helpers):

    df, spark, id_col = dataframe

    if spark:
        # i.e. Spark DataFrame -> Spark list[Row]
        df = df.collect()

    tfidf = StrStartsWith(**input)
    tfidf.with_frame(_wrap(df, id_col))

    df = tfidf.dedupe("address").unwrap()

    assert helpers.get_column_as_list(df, GROUP_ID) == output


star_ends_parametrize_data = [
    # i.e. no deduping because no string starts with the pattern
    ({"pattern": "zzzzz", "case": True}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    ({"pattern": "zzzzz", "case": False}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # String does dedupe if case correct; but doesn't otherwise
    ({"pattern": "kingdom", "case": True}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    ({"pattern": "kingdom", "case": False}, [1, 2, 3, 4, 2, 2, 7, 8, 9, 10, 11, 12, 13]),
]


@pytest.mark.parametrize("input, output", star_ends_parametrize_data)
def test_str_ends_dedupe_integrated(input, output, dataframe, helpers):

    df, spark, id_col = dataframe

    if spark:
        # i.e. Spark DataFrame -> Spark list[Row]
        df = df.collect()

    tfidf = StrEndsWith(**input)
    tfidf.with_frame(_wrap(df, id_col))

    df = tfidf.dedupe("address").unwrap()

    assert helpers.get_column_as_list(df, GROUP_ID) == output


star_contains_parametrize_data = [
    # i.e. no deduping because no string starts with the pattern
    ({"pattern": "zzzzz", "case": True, "regex": True}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    ({"pattern": "zzzzz", "case": False, "regex": True}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    ({"pattern": "zzzzz", "case": True, "regex": False}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    ({"pattern": "zzzzz", "case": False, "regex": False}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # String does dedupe if case correct; but doesn't otherwise, no regex
    ({"pattern": "ol5 9pl", "case": True, "regex": False}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    ({"pattern": "ol5 9pl", "case": False, "regex": False}, [1, 2, 3, 4, 5, 1, 7, 8, 1, 1, 11, 12, 13]),
    # String does dedupe if case correct; but doesn't otherwise, with regex
    ({"pattern": r"05\d{3}", "case": True, "regex": True}, [1, 2, 3, 3, 5, 6, 3, 8, 9, 10, 11, 12, 13]),
    ({"pattern": r"05\d{3}", "case": False, "regex": True}, [1, 2, 3, 3, 5, 6, 3, 8, 9, 10, 11, 12, 13]),
]


@pytest.mark.parametrize("input, output", star_contains_parametrize_data)
def test_str_contains_dedupe_integrated(input, output, dataframe, helpers):

    df, spark, id_col = dataframe

    if spark:
        # i.e. Spark DataFrame -> Spark list[Row]
        df = df.collect()

    tfidf = StrContains(**input)
    tfidf.with_frame(_wrap(df, id_col))

    df = tfidf.dedupe("address").unwrap()

    assert helpers.get_column_as_list(df, GROUP_ID) == output
