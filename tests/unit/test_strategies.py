"""Tests for dupegrouper.strategies"""

from __future__ import annotations
from unittest.mock import Mock

import numpy as np
import pytest

from dupegrouper.base import wrap
from dupegrouper.definitions import CANONICAL_ID
from dupegrouper.strategies import BaseStrategy, ThresholdDedupers
from dupegrouper.dataframe import WrappedDataFrame

@pytest.fixture
def base_strategy_stub():
    strat = Mock(spec=BaseStrategy)

    strat.rule = "first"
    strat.validate = Mock()
    strat.get_array = Mock(return_value=[1, 2])
    strat._gen_similarity_pairs = Mock(return_value=[(0, 1)])
    strat._get_components.return_value = {0: [0], 1: [1]}

    return strat

@pytest.fixture
def wrapped_df_stub(base_strategy_stub):
    wrapped = Mock()
    wrapped.get_col.return_value = [1, 2]
    wrapped.put_col.return_value = wrapped

    base_strategy_stub.wrapped_df = wrapped
    base_strategy_stub._get_components.return_value = {0: [0, 1]}

    return wrapped

def test_canonicalize_calls__get_components(base_strategy_stub, wrapped_df_stub):
    BaseStrategy.canonicalize(base_strategy_stub, "col")
    base_strategy_stub._get_components.assert_called_once_with("col")
    wrapped_df_stub.put_col.assert_called_once()


def test_canonicalize_uses_rule_first(base_strategy_stub, wrapped_df_stub):
    base_strategy_stub.rule = "first"
    BaseStrategy.canonicalize(base_strategy_stub, "col") 
    args, _ = wrapped_df_stub.put_col.call_args
    assert args[0] == CANONICAL_ID
    np.testing.assert_array_equal(args[1], np.array([1, 1]))

def test_canonicalize_uses_rule_last(base_strategy_stub, wrapped_df_stub):
    base_strategy_stub.rule = "last"
    BaseStrategy.canonicalize(base_strategy_stub, "col")
    args, _ = wrapped_df_stub.put_col.call_args
    assert args[0] == CANONICAL_ID
    np.testing.assert_array_equal(args[1], np.array([2, 2]))

def test_get_components_calls_validate_and_gen_pairs(base_strategy_stub):
    
    base_strategy_stub.get_array.return_value = [0, 1, 2]
    base_strategy_stub._gen_similarity_pairs.return_value = [(0, 2)]

    components = BaseStrategy._get_components(base_strategy_stub, "col")

    base_strategy_stub.validate.assert_called_once_with("col")
    base_strategy_stub._gen_similarity_pairs.assert_called_once()
    assert components == {0: [0, 2], 1: [1]}

def test_threshold_validation():
    with pytest.raises(ValueError):
        ThresholdDedupers(threshold=1.0)

def test_reinstantiate():
    dummy_positional_args = ("dummy", False)
    dummy_kwargs = {"test": 5, "random": "random_arg"}

    instance = BaseStrategy(*dummy_positional_args, **dummy_kwargs)

    instance_reinstantiated = instance.reinstantiate()

    assert instance is not instance_reinstantiated
    assert instance._init_args == instance_reinstantiated._init_args == dummy_positional_args
    assert instance._init_kwargs == instance_reinstantiated._init_kwargs == dummy_kwargs


def test_bind_frame(dataframe):

    df, _, _ = dataframe

    strategy = BaseStrategy()
    strategy.bind_frame(wrap(df))

    assert isinstance(strategy.wrapped_df, WrappedDataFrame)

def test_bind_rule():

    strategy = BaseStrategy()
    strategy.bind_rule("first")

    assert strategy.rule == "first"

    with pytest.raises(ValueError):
        strategy.bind_rule("random")
    


# def test_custom_canonicalize(df_pandas):

#     canonicalizer = Custom(my_func, "address", match_str="navarra")
#     canonicalizer.bind_frame(wrap(df_pandas)).bind_rule("first")

#     updatedwrapped_df = canonicalizer.canonicalize()
#     updated_df = updatedwrapped_df.unwrap()

#     expected_canonical_ids = [1, 2, 3, 3, 5, 6, 3, 8, 1, 1, 11, 12, 13]

#     assert list(updated_df["canonical_id"]) == expected_canonical_ids