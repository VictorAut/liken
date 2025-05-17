from __future__ import annotations
from unittest.mock import Mock, patch, call

import numpy as np
import pytest

from dupegrouper.base import _wrap
from dupegrouper.definitions import TMP_ATTR, GROUP_ID
from dupegrouper.strategies.fuzzy import Fuzzy


####################
# DEDUPE UNIT TEST #
####################


def test_dedupe_unit():
    attr = "address"
    dummy_array = np.array(["foo", "bar", "bar"])

    tfidf = Fuzzy(tolerance=0.2)

    mock_wrapped_df = Mock()
    mock_wrapped_df.get_col.return_value = dummy_array
    tfidf.wrapped_df = mock_wrapped_df

    with patch.object(
        tfidf,
        "_fuzz_ratio",
        return_value=85.1,
    ) as mock_fuzz, patch.object(
        tfidf,
        "assign_group_id",
        return_value=mock_wrapped_df,
    ) as mock_assign_group_id:

        # Also mock wrapped_df chaining methods
        mock_wrapped_df.map_dict.return_value = [None, "bar", "bar"]
        mock_wrapped_df.put_col.return_value = mock_wrapped_df
        mock_wrapped_df.assign_group_id.return_value = mock_wrapped_df
        mock_wrapped_df.drop_col.return_value = mock_wrapped_df

        # Run dedupe
        result = tfidf.dedupe(attr)

        # Assertions
        mock_fuzz.assert_called_with("foo", "foo")

        mock_wrapped_df.map_dict.assert_called_once_with(attr, {"bar": "foo", "foo": "foo"})

        # second put call is part of assign_group_id which in another unit test
        put_col_call = mock_wrapped_df.put_col.call_args_list[0]
        assert put_col_call == call(TMP_ATTR, [None, "bar", "bar"])

        mock_assign_group_id.assert_called_once()
        mock_wrapped_df.drop_col.assert_called_once()

        assert result == mock_wrapped_df


##################################
# DEDUPE NARROW INTEGRATION TEST #
##################################


fuzzy_parametrize_data = [
    # i.e. no deduping
    ({"tolerance": 0}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # progressive deduping
    ({"tolerance": 0.05}, [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 11, 12, 13]),
    ({"tolerance": 0.15}, [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 11, 12, 13]),
    ({"tolerance": 0.25}, [1, 2, 3, 3, 5, 6, 7, 8, 1, 1, 11, 12, 13]),
    ({"tolerance": 0.35}, [1, 2, 3, 3, 5, 6, 7, 2, 1, 1, 11, 12, 13]),
    ({"tolerance": 0.45}, [1, 2, 3, 3, 5, 6, 3, 2, 1, 1, 11, 12, 13]),
    ({"tolerance": 0.55}, [1, 2, 3, 3, 5, 5, 3, 2, 1, 1, 5, 1, 13]),
    ({"tolerance": 0.65}, [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 2, 12, 13]),
    ({"tolerance": 0.75}, [1, 2, 3, 3, 3, 3, 3, 3, 1, 1, 3, 12, 3]),
    ({"tolerance": 0.85}, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 12, 1]),
    ({"tolerance": 0.95}, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
]


@pytest.mark.parametrize("input, output", fuzzy_parametrize_data)
def test_dedupe_integrated(input, output, dataframe, helpers):

    df, spark, id_col = dataframe

    if spark:
        # i.e. Spark DataFrame -> Spark list[Row]
        df = df.collect()

    tfidf = Fuzzy(**input)
    tfidf.with_frame(_wrap(df, id_col))

    df = tfidf.dedupe("address").unwrap()

    assert helpers.get_column_as_list(df, GROUP_ID) == output
