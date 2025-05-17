"""Tests for dupegrouper.strategy"""

from __future__ import annotations
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from dupegrouper.base import _wrap
from dupegrouper.definitions import GROUP_ID
from dupegrouper.strategy import DeduplicationStrategy
from dupegrouper.wrappers import WrappedDataFrame


class DummyStrategy(DeduplicationStrategy):
    def dedupe(self, attr: str):
        return self.assign_group_id(attr).unwrap()


###########################
# `DeduplicationStrategy` #
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
    strategy.with_frame(_wrap(df))

    assert isinstance(strategy.wrapped_df, WrappedDataFrame)


@pytest.mark.parametrize(
    "attribute_array, expected_group_id",
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
def test_assign_group_id(attribute_array, expected_group_id):
    attr = "address"
    input_group_ids = [1, 2, 3, 4, 5, 6]

    mock_wrapped_df = Mock()
    mock_wrapped_df.get_col.side_effect = lambda key: attribute_array if key == attr else input_group_ids
    mock_wrapped_df.put_col.return_value = expected_group_id

    class Dummy(DeduplicationStrategy):
        def __init__(self, wrapped_df):
            self.wrapped_df = wrapped_df

        def dedupe():  # ABC contract forces this
            pass

    obj = Dummy(mock_wrapped_df)
    result = obj.assign_group_id(attr)

    # Assert
    mock_wrapped_df.get_col.assert_any_call(attr)
    mock_wrapped_df.get_col.assert_any_call(GROUP_ID)
    mock_wrapped_df.put_col.assert_called_once()
    np.testing.assert_array_equal(result, expected_group_id)


def test_dedupe(helpers):
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
            "group_id": [1, 2, 3, 4, 5, 6],
        }
    )

    strategy = DummyStrategy()
    strategy.with_frame(_wrap(df))

    deduped_df = strategy.dedupe("name")  # Uses assign_group_id internally

    expected_groups = [1, 2, 1, 4, 2, 4]
    assert helpers.get_column_as_list(deduped_df, GROUP_ID) == expected_groups
