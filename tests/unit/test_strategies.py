"""Tests for dupegrouper.strategies"""

from __future__ import annotations
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from dupegrouper.base import wrap
from dupegrouper.definitions import CANONICAL_ID
from dupegrouper.strategies import BaseStrategy
from dupegrouper.dataframe import WrappedDataFrame


class DummyStrategy(BaseStrategy):
    def canonicalize(self, attr: str):
        return self.propagate_canonical_id(attr).unwrap()


###########################
# `BaseStrategy` #
###########################


def test_reinstantiate():
    dummy_positional_args = ("dummy", False)
    dummy_kwargs = {"test": 5, "random": "random_arg"}

    instance = DummyStrategy(*dummy_positional_args, **dummy_kwargs)

    instance_reinstantiated = instance.reinstantiate()

    assert instance is not instance_reinstantiated
    assert instance._init_args == instance_reinstantiated._init_args == dummy_positional_args
    assert instance._init_kwargs == instance_reinstantiated._init_kwargs == dummy_kwargs


def test_with_frame(dataframe):

    df, _, _ = dataframe

    strategy = DummyStrategy()
    strategy.with_frame(wrap(df))

    assert isinstance(strategy.wrapped_df, WrappedDataFrame)


@pytest.mark.parametrize(
    "attribute_array, expected_canonical_id",
    [
        # standard: matches
        (["Alice", "Bob", "Alice", "Charlie", "Bob", "Charlie"], [1, 2, 1, 4, 2, 4]),
        # Mixed casing: no matches
        (["Alice", "Bob", "alice", "charlie", "Bob", "Charlie"], [1, 2, 3, 4, 2, 6]),
        # int numbers
        ([111, 123, 321, 999, 654, 999], [1, 2, 3, 4, 5, 4]),
        # floats numbers
        ([111.0, 123.0, 321.0, 999.0, 654.0, 999.0], [1, 2, 3, 4, 5, 4]),
        # mixed numbers
        ([111.0, 123.0, 321.0, 999, 654.0, 999.0], [1, 2, 3, 4, 5, 4]),
        # white space: no matches
        (["Alice", "Bob", "Alice     ", "Charlie", "   Bob", "Charlie"], [1, 2, 3, 4, 5, 4]),
    ],
    ids=[
        "string matches",
        "no string matches",
        "int matches",
        "float matches",
        "mixed numeric matches",
        "whitespace no string match",
    ],
)
def test_propagate_canonical_id(attribute_array, expected_canonical_id):
    attr = "address"
    input_canonical_ids = [1, 2, 3, 4, 5, 6]

    mockwrapped_df = Mock()
    mockwrapped_df.get_col.side_effect = lambda key: attribute_array if key == attr else input_canonical_ids
    mockwrapped_df.put_col.return_value = expected_canonical_id

    class Dummy(BaseStrategy):
        def __init__(self, wrapped_df):
            self.wrapped_df = wrapped_df

        def canonicalize():  # ABC contract forces this
            pass

    obj = Dummy(mockwrapped_df)
    result = obj.propagate_canonical_id(attr)

    # Assert
    mockwrapped_df.get_col.assert_any_call(attr)
    mockwrapped_df.get_col.assert_any_call(CANONICAL_ID)
    mockwrapped_df.put_col.assert_called_once()
    np.testing.assert_array_equal(result, expected_canonical_id)


def test_canonicalize(helpers):
    """In a way, this essentially mimics testing `dupegrouper.strategies.Exact`"""

    df = pd.DataFrame(
        {
            "name": [
                "Alice",
                "Bob",
                "Alice",
                "Charlie",
                "Bob",
                "Charlie",
            ],
            "canonical_id": [1, 2, 3, 4, 5, 6],
        }
    )

    strategy = DummyStrategy()
    strategy.with_frame(wrap(df))

    canonicalized_df = strategy.canonicalize("name")  # Uses propagate_canonical_id internally

    expected_groups = [1, 2, 1, 4, 2, 4]
    assert helpers.get_column_as_list(canonicalized_df, CANONICAL_ID) == expected_groups


############################################

import pandas as pd

from dupegrouper.base import wrap
from dupegrouper.strategies import Custom


# Custom callable function
def my_func(df: pd.DataFrame, attr: str, /, match_str: str) -> dict[str, str]:
    my_map = {}
    for irow, _ in df.iterrows():
        left: str = df.at[irow, attr]
        my_map[left] = left
        for jrow, _ in df.iterrows():
            right: str = df.at[jrow, attr]
            if match_str in left.lower() and match_str in right.lower():
                my_map[left] = right
                break
    return my_map


def test_custom_canonicalize(df_pandas):

    canonicalizer = Custom(my_func, "address", match_str="navarra")
    canonicalizer.with_frame(wrap(df_pandas))

    updatedwrapped_df = canonicalizer.canonicalize()
    updated_df = updatedwrapped_df.unwrap()

    expected_canonical_ids = [1, 2, 3, 3, 5, 6, 3, 8, 1, 1, 11, 12, 13]

    assert list(updated_df["canonical_id"]) == expected_canonical_ids


from unittest.mock import Mock, patch, call

import numpy as np
import pytest

from dupegrouper.base import wrap
from dupegrouper.definitions import TMP_ATTR_LABEL, CANONICAL_ID
from dupegrouper.strategies import Fuzzy


####################
# DEDUPE UNIT TEST #
####################


# def test_canonicalize_unit():
#     attr = "address"
#     dummy_array = np.array(["foo", "bar", "bar"])

#     tfidf = Fuzzy(threshold=0.2)

#     mockwrapped_df = Mock()clear

#     mockwrapped_df.get_col.return_value = dummy_array
#     tfidf.wrapped_df = mockwrapped_df

#     with patch.object(
#         tfidf,
#         "_fuzz_ratio",
#         return_value=85.1,
#     ) as mock_fuzz, patch.object(
#         tfidf,
#         "propagate_canonical_id",
#         return_value=mockwrapped_df,
#     ) as mock_propagate_canonical_id:

#         # Also mock wrapped_df chaining methods
#         mockwrapped_df.map_dict.return_value = [None, "bar", "bar"]
#         mockwrapped_df.put_col.return_value = mockwrapped_df
#         mockwrapped_df.propagate_canonical_id.return_value = mockwrapped_df
#         mockwrapped_df.drop_col.return_value = mockwrapped_df

#         # Run canonicalize
#         result = tfidf.canonicalize(attr)

#         # Assertions
#         mock_fuzz.assert_called_with("foo", "foo")

#         mockwrapped_df.map_dict.assert_called_once_with(attr, {"bar": "foo", "foo": "foo"})

#         # second put call is part of propagate_canonical_id which in another unit test
#         put_col_call = mockwrapped_df.put_col.call_args_list[0]
#         assert put_col_call == call(TMP_ATTR_LABEL, [None, "bar", "bar"])

#         mock_propagate_canonical_id.assert_called_once()
#         mockwrapped_df.drop_col.assert_called_once()

#         assert result == mockwrapped_df


from unittest.mock import Mock, patch, call

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

from dupegrouper.base import wrap
from dupegrouper.definitions import TMP_ATTR_LABEL, CANONICAL_ID
from dupegrouper.strategies import TfIdf



####################
# DEDUPE UNIT TEST #
####################


# def test_canonicalize_unit():
#     attr = "address"
#     dummy_array = np.array(["foo", "bar", "bar"])

#     tfidf = TfIdf(ngram=(1, 1), threshold=0.2, topn=2)

#     # mock for wrapped_df
#     mockwrapped_df = Mock()
#     mockwrapped_df.get_col.return_value = dummy_array
#     tfidf.wrapped_df = mockwrapped_df

#     with patch.object(tfidf, "_vectorize", return_value="dummy-vectorizer") as mock_vec, patch.object(
#         tfidf, "_get_similarities_matrix", return_value="dummy-matrix"
#     ) as mock_sim_matrix, patch.object(
#         tfidf, "_get_matches_array", return_value=(np.array([0]), np.array([1]), np.array([0.95]))
#     ) as mock_matches_array, patch.object(
#         tfidf, "_gen_map", return_value=iter([{"bar": "bar"}])
#     ) as mock_gen_map, patch.object(
#         tfidf, "propagate_canonical_id", return_value=mockwrapped_df
#     ) as mock_propagate_canonical_id:

#         mockwrapped_df.map_dict.return_value = [None, "bar", "bar"]
#         mockwrapped_df.fill_na.return_value = ["foo", "bar", "bar"]
#         mockwrapped_df.put_col.return_value = mockwrapped_df
#         mockwrapped_df.propagate_canonical_id.return_value = mockwrapped_df
#         mockwrapped_df.drop_col.return_value = mockwrapped_df

#         result = tfidf.canonicalize(attr)

#         # Assertions
#         mock_vec.assert_called_once_with((1, 1))

#         args, _ = mock_sim_matrix.call_args
#         assert args[0] == "dummy-vectorizer"
#         np.testing.assert_array_equal(args[1], dummy_array)

#         args, _ = mock_matches_array.call_args
#         assert args[0] == "dummy-matrix"
#         np.testing.assert_array_equal(args[1], dummy_array)

#         mock_gen_map.assert_called_once()

#         mockwrapped_df.map_dict.assert_called_once_with(attr, {"bar": "bar"})
#         mockwrapped_df.fill_na.assert_called_once()

#         # second put call is part of propagate_canonical_id which in another unit test
#         put_col_call = mockwrapped_df.put_col.call_args_list[0]
#         assert put_col_call == call(TMP_ATTR_LABEL, ["foo", "bar", "bar"])

#         mock_propagate_canonical_id.assert_called_once()
#         mockwrapped_df.drop_col.assert_called_once()

#         assert result == mockwrapped_df


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


def test_str_starts_canonicalize_unit():
    attr = "address"
    dummy_array = np.array(["foo", "bar", "bar"])

    canonicalizer = StrStartsWith(pattern="b")

    mockwrapped_df = Mock()
    mockwrapped_df.get_col.return_value = dummy_array
    canonicalizer.wrapped_df = mockwrapped_df

    with patch.object(
        canonicalizer,
        "propagate_canonical_id",
        return_value=mockwrapped_df,
    ) as mock_propagate_canonical_id:

        # Also mock wrapped_df chaining methods
        mockwrapped_df.map_dict.return_value = [None, "bar", "bar"]
        mockwrapped_df.put_col.return_value = mockwrapped_df
        mockwrapped_df.propagate_canonical_id.return_value = mockwrapped_df
        mockwrapped_df.drop_col.return_value = mockwrapped_df

        # Run canonicalize
        result = canonicalizer.canonicalize(attr)

        mockwrapped_df.map_dict.assert_called_once_with(attr, {"bar": "bar"})

        # second put call is part of propagate_canonical_id which in another unit test
        put_col_call = mockwrapped_df.put_col.call_args_list[0]
        assert put_col_call == call(TMP_ATTR_LABEL, [None, "bar", "bar"])

        mock_propagate_canonical_id.assert_called_once()
        mockwrapped_df.drop_col.assert_called_once()

        assert result == mockwrapped_df


def test_str_ends_canonicalize_unit():
    attr = "address"
    dummy_array = np.array(["foo", "bar", "bar"])

    canonicalizer = StrEndsWith(pattern="ar")

    mockwrapped_df = Mock()
    mockwrapped_df.get_col.return_value = dummy_array
    canonicalizer.wrapped_df = mockwrapped_df

    with patch.object(
        canonicalizer,
        "propagate_canonical_id",
        return_value=mockwrapped_df,
    ) as mock_propagate_canonical_id:

        # Also mock wrapped_df chaining methods
        mockwrapped_df.map_dict.return_value = [None, "bar", "bar"]
        mockwrapped_df.put_col.return_value = mockwrapped_df
        mockwrapped_df.propagate_canonical_id.return_value = mockwrapped_df
        mockwrapped_df.drop_col.return_value = mockwrapped_df

        # Run canonicalize
        result = canonicalizer.canonicalize(attr)

        mockwrapped_df.map_dict.assert_called_once_with(attr, {"bar": "bar"})

        # second put call is part of propagate_canonical_id which in another unit test
        put_col_call = mockwrapped_df.put_col.call_args_list[0]
        assert put_col_call == call(TMP_ATTR_LABEL, [None, "bar", "bar"])

        mock_propagate_canonical_id.assert_called_once()
        mockwrapped_df.drop_col.assert_called_once()

        assert result == mockwrapped_df


def test_str_contains_canonicalize_unit():
    attr = "address"
    dummy_array = np.array(["foo", "bar", "bar"])

    canonicalizer = StrContains(pattern="a")

    mockwrapped_df = Mock()
    mockwrapped_df.get_col.return_value = dummy_array
    canonicalizer.wrapped_df = mockwrapped_df

    with patch.object(
        canonicalizer,
        "propagate_canonical_id",
        return_value=mockwrapped_df,
    ) as mock_propagate_canonical_id:

        # Also mock wrapped_df chaining methods
        mockwrapped_df.map_dict.return_value = [None, "bar", "bar"]
        mockwrapped_df.put_col.return_value = mockwrapped_df
        mockwrapped_df.propagate_canonical_id.return_value = mockwrapped_df
        mockwrapped_df.drop_col.return_value = mockwrapped_df

        # Run canonicalize
        result = canonicalizer.canonicalize(attr)

        mockwrapped_df.map_dict.assert_called_once_with(attr, {"bar": "bar"})

        # second put call is part of propagate_canonical_id which in another unit test
        put_col_call = mockwrapped_df.put_col.call_args_list[0]
        assert put_col_call == call(TMP_ATTR_LABEL, [None, "bar", "bar"])

        mock_propagate_canonical_id.assert_called_once()
        mockwrapped_df.drop_col.assert_called_once()

        assert result == mockwrapped_df