from __future__ import annotations
from unittest.mock import Mock, patch, call

import numpy as np
import pytest

from dupegrouper.base import wrap
from dupegrouper.definitions import TMP_ATTR_LABEL, CANONICAL_ID
from dupegrouper.strategies import (
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

    mockwrapped_df = Mock()
    mockwrapped_df.get_col.return_value = dummy_array
    deduper.wrapped_df = mockwrapped_df

    with patch.object(
        deduper,
        "canonicalize",
        return_value=mockwrapped_df,
    ) as mock_canonicalize:

        # Also mock wrapped_df chaining methods
        mockwrapped_df.map_dict.return_value = [None, "bar", "bar"]
        mockwrapped_df.put_col.return_value = mockwrapped_df
        mockwrapped_df.canonicalize.return_value = mockwrapped_df
        mockwrapped_df.drop_col.return_value = mockwrapped_df

        # Run dedupe
        result = deduper.dedupe(attr)

        mockwrapped_df.map_dict.assert_called_once_with(attr, {"bar": "bar"})

        # second put call is part of canonicalize which in another unit test
        put_col_call = mockwrapped_df.put_col.call_args_list[0]
        assert put_col_call == call(TMP_ATTR_LABEL, [None, "bar", "bar"])

        mock_canonicalize.assert_called_once()
        mockwrapped_df.drop_col.assert_called_once()

        assert result == mockwrapped_df


def test_str_ends_dedupe_unit():
    attr = "address"
    dummy_array = np.array(["foo", "bar", "bar"])

    deduper = StrEndsWith(pattern="ar")

    mockwrapped_df = Mock()
    mockwrapped_df.get_col.return_value = dummy_array
    deduper.wrapped_df = mockwrapped_df

    with patch.object(
        deduper,
        "canonicalize",
        return_value=mockwrapped_df,
    ) as mock_canonicalize:

        # Also mock wrapped_df chaining methods
        mockwrapped_df.map_dict.return_value = [None, "bar", "bar"]
        mockwrapped_df.put_col.return_value = mockwrapped_df
        mockwrapped_df.canonicalize.return_value = mockwrapped_df
        mockwrapped_df.drop_col.return_value = mockwrapped_df

        # Run dedupe
        result = deduper.dedupe(attr)

        mockwrapped_df.map_dict.assert_called_once_with(attr, {"bar": "bar"})

        # second put call is part of canonicalize which in another unit test
        put_col_call = mockwrapped_df.put_col.call_args_list[0]
        assert put_col_call == call(TMP_ATTR_LABEL, [None, "bar", "bar"])

        mock_canonicalize.assert_called_once()
        mockwrapped_df.drop_col.assert_called_once()

        assert result == mockwrapped_df


def test_str_contains_dedupe_unit():
    attr = "address"
    dummy_array = np.array(["foo", "bar", "bar"])

    deduper = StrContains(pattern="a")

    mockwrapped_df = Mock()
    mockwrapped_df.get_col.return_value = dummy_array
    deduper.wrapped_df = mockwrapped_df

    with patch.object(
        deduper,
        "canonicalize",
        return_value=mockwrapped_df,
    ) as mock_canonicalize:

        # Also mock wrapped_df chaining methods
        mockwrapped_df.map_dict.return_value = [None, "bar", "bar"]
        mockwrapped_df.put_col.return_value = mockwrapped_df
        mockwrapped_df.canonicalize.return_value = mockwrapped_df
        mockwrapped_df.drop_col.return_value = mockwrapped_df

        # Run dedupe
        result = deduper.dedupe(attr)

        mockwrapped_df.map_dict.assert_called_once_with(attr, {"bar": "bar"})

        # second put call is part of canonicalize which in another unit test
        put_col_call = mockwrapped_df.put_col.call_args_list[0]
        assert put_col_call == call(TMP_ATTR_LABEL, [None, "bar", "bar"])

        mock_canonicalize.assert_called_once()
        mockwrapped_df.drop_col.assert_called_once()

        assert result == mockwrapped_df


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
    tfidf.with_frame(wrap(df, id_col))

    df = tfidf.dedupe("address").unwrap()

    assert helpers.get_column_as_list(df, CANONICAL_ID) == output


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
    tfidf.with_frame(wrap(df, id_col))

    df = tfidf.dedupe("address").unwrap()

    assert helpers.get_column_as_list(df, CANONICAL_ID) == output


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
    tfidf.with_frame(wrap(df, id_col))

    df = tfidf.dedupe("address").unwrap()

    assert helpers.get_column_as_list(df, CANONICAL_ID) == output
