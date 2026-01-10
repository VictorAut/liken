from unittest.mock import Mock, patch

import pytest

from dupegrouper.strats_manager import (
    StratsConfig,
    StrategyManager,
    StrategyConfigTypeError,
)
from dupegrouper.strats_library import BaseStrategy
from dupegrouper.constants import DEFAULT_STRAT_KEY


###########
# Helpers #
###########


class DummyStrategy(BaseStrategy):
    pass


@pytest.fixture
def s1():
    return DummyStrategy()


@pytest.fixture
def s2():
    return DummyStrategy()


@pytest.fixture
def s3():
    return DummyStrategy()


#####################
# StratsConfig tests
#####################


@pytest.mark.parametrize(
    "columns, strat",
    [
        ("address", [BaseStrategy()]),
        ("address", (BaseStrategy(),)),
        (("address", "email"), [BaseStrategy()]),
        (("address", "email"), (BaseStrategy(),)),
    ],
)
def test_stratsconfig_accepts_inputs(columns, strat):
    config = StratsConfig()
    config[columns] = strat

    assert columns in config
    assert config[columns] == strat


def test_stratsconfig_rejects_invalid_key_type(s1):
    config = StratsConfig()
    with pytest.raises(StrategyConfigTypeError, match="Invalid type for strat dict key"):
        config[123] = [s1]


def test_stratsconfig_rejects_invalid_value_type():
    config = StratsConfig()
    with pytest.raises(StrategyConfigTypeError, match="Invalid type for strat dict value"):
        config["col"] = "not_a_strategy"


def test_stratsconfig_rejects_invalid_member_in_value(s1, s2, s3):
    config = StratsConfig()
    with pytest.raises(StrategyConfigTypeError, match="Invalid type for strat dict value member"):
        config["col"] = [s1, "bad", s2, s3]


################
# apply method #
################


def test_strategy_manager_apply_single_strategy_once(s1):
    sm = StrategyManager()
    sm.apply(s1)

    strats = sm.get()
    assert s1 in strats[DEFAULT_STRAT_KEY]


def test_strategy_manager_apply_single_strategy_multiple(s1, s2, s3):
    sm = StrategyManager()
    sm.apply(s1)
    sm.apply(s2)
    sm.apply(s3)

    strats = sm.get()
    assert s1 in strats[DEFAULT_STRAT_KEY]
    assert s2 in strats[DEFAULT_STRAT_KEY]
    assert s3 in strats[DEFAULT_STRAT_KEY]


def test_strategy_manager_apply_dict(s1, s2, s3):
    sm = StrategyManager()
    custom = {"a": [s1], "b": (s2, s3)}

    sm.apply(custom)
    result = sm.get()

    assert result["a"] == [s1]
    assert result["b"] == (s2, s3)


def test_strategy_manager_apply_stratsconfig(s1, s2, s3):
    sm = StrategyManager()
    config = StratsConfig({"a": [s1], "b": (s2, s3)})

    sm.apply(config)
    result = sm.get()

    assert result["a"] == [s1]
    assert result["b"] == (s2, s3)


def test_strategy_manager_apply_invalid_type():
    sm = StrategyManager()
    with pytest.raises(StrategyConfigTypeError):
        sm.apply(123)


#######
# get #
#######


def test_strategy_manager_get_returns_config():
    sm = StrategyManager()
    result = sm.get()
    assert isinstance(result, StratsConfig)


##############
# pretty_get #
##############


def test_pretty_get_single_default_key(s1, s2):
    sm = StrategyManager()
    sm.apply(s1)
    sm.apply(s2)

    pretty = sm.pretty_get()
    assert pretty == ("DummyStrategy", "DummyStrategy")


def test_pretty_get_multiple_keys(s1, s2, s3):
    sm = StrategyManager()
    sm.apply({"col_a": [s1, s3], "col_b": [s2]})

    pretty = sm.pretty_get()
    assert pretty == {
        "col_a": ("DummyStrategy", "DummyStrategy"),
        "col_b": ("DummyStrategy",),
    }


#########
# reset #
#########


def test_strategy_manager_reset_clears_strats(s1):
    sm = StrategyManager()
    sm.apply(s1)

    sm.reset()
    assert sm.get() == {}
    assert sm.pretty_get() is None


###########################
# StrategyConfigTypeError #
###########################


def test_strategy_config_type_error_is_type_error():
    err = StrategyConfigTypeError("bad")
    assert isinstance(err, TypeError)
