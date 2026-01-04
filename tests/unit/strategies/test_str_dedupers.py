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

