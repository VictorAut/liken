
import pytest

from enlace._strats_library import BaseStrategy
from enlace._strats_manager import (
    SEQUENTIAL_API_DEFAULT_KEY,
    InvalidStrategyError,
    On,
    Rules,
    StrategyManager,
    StratsDict,
    on,
)


###########
# Helpers #
###########


class DummyStrategy(BaseStrategy):
    def __str__(self):
        return self.str_representation("dummy_strategy")


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
# StratsDict tests
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
    config = StratsDict()
    config[columns] = strat

    assert columns in config
    assert config[columns] == strat


def test_stratsconfig_rejects_invalid_key_type(s1):
    config = StratsDict()
    with pytest.raises(InvalidStrategyError, match="Invalid type for dict key type"):
        config[123] = [s1]


def test_stratsconfig_rejects_invalid_value_type():
    config = StratsDict()
    with pytest.raises(InvalidStrategyError, match="Invalid type for dict value"):
        config["col"] = "not_a_strategy"


def test_stratsconfig_rejects_invalid_member_in_value(s1, s2, s3):
    config = StratsDict()
    with pytest.raises(InvalidStrategyError, match="Invalid type for dict value member"):
        config["col"] = [s1, "bad", s2, s3]


################
# apply method #
################


def test_strategy_manager_apply_sequential_once(s1):
    sm = StrategyManager()
    sm.apply(s1)

    strats = sm.get()
    assert s1 in strats[SEQUENTIAL_API_DEFAULT_KEY]


def test_strategy_manager_apply_sequential_multiple(s1, s2, s3):
    sm = StrategyManager()
    sm.apply(s1)
    sm.apply(s2)
    sm.apply(s3)

    strats = sm.get()
    assert s1 in strats[SEQUENTIAL_API_DEFAULT_KEY]
    assert s2 in strats[SEQUENTIAL_API_DEFAULT_KEY]
    assert s3 in strats[SEQUENTIAL_API_DEFAULT_KEY]


def test_strategy_manager_apply_dict_single(s1, s2, s3):
    sm = StrategyManager()
    strat = {"a": [s1], "b": (s2), "c": s3}

    sm.apply(strat)
    result = sm.get()

    assert result["a"] == [s1]
    assert result["b"] == (s2,)
    assert result["c"] == (s3,)


def test_strategy_manager_apply_dict(s1, s2, s3):
    sm = StrategyManager()
    strat = {"a": [s1], "b": (s2, s3)}

    sm.apply(strat)
    result = sm.get()

    assert result["a"] == [s1]
    assert result["b"] == (s2, s3)


def test_strategy_manager_apply_single_on_not_rule(s1):
    sm = StrategyManager()
    strat = on("a", s1)  # Not a tuple!

    sm.apply(strat)
    result = sm.get()

    assert isinstance(result, tuple)
    assert len(result) == 1
    assert isinstance(result[0], On)
    assert result[0]._column == "a"
    assert isinstance(result[0]._strat, type(s1))


def test_strategy_manager_apply_single_on_as_tuple(s1):
    sm = StrategyManager()
    strat = (on("a", s1),)  # Is a tuple

    sm.apply(strat)
    result = sm.get()

    assert isinstance(result, tuple)
    assert len(result) == 1
    assert isinstance(result[0], On)
    assert result[0]._column == "a"
    assert isinstance(result[0]._strat, type(s1))


def test_strategy_manager_apply_single_on_as_rule(s1):
    sm = StrategyManager()
    strat = Rules(
        on("a", s1),
    )

    sm.apply(strat)
    result = sm.get()

    assert isinstance(result, tuple)
    assert len(result) == 1
    assert isinstance(result[0], On)
    assert result[0]._column == "a"
    assert isinstance(result[0]._strat, type(s1))


def test_strategy_manager_apply_tuple(s1, s2, s3):
    sm = StrategyManager()
    strat = Rules(on("a", s1), on("b", s2) & on("c", s3))

    sm.apply(strat)
    result = sm.get()

    assert isinstance(result, tuple)
    assert len(result) == 2  # As the & operator returns instance of self.


def test_strategy_manager_apply_rejects_invalid_type():
    sm = StrategyManager()
    with pytest.raises(InvalidStrategyError, match="Invalid strategy"):
        sm.apply(123)


def test_strategy_manager_apply_rejects_sequence_after_dict(s1, s2, s3):
    sm = StrategyManager()
    sm.apply({"a": (s1,), "b": (s1, s2)})  # legal
    with pytest.raises(InvalidStrategyError, match="Cannot apply a 'BaseStrategy' after a strategy mapping"):
        sm.apply(s3)


def test_strategy_manager_apply_warns_dict_after_sequence():
    sm = StrategyManager()
    strat = BaseStrategy()

    sm.apply(strat)

    with pytest.warns(
        UserWarning,
        match="Replacing previously added sequence strategy with a dict strategy",
    ):
        sm.apply({"email": [strat]})


def test_rule_rejects_empty():
    with pytest.raises(InvalidStrategyError, match="Rules cannot be empty"):
        Rules()


def test_rule_rejects_not_instance_of_on():
    with pytest.raises(InvalidStrategyError, match="Invalid Rules element at index"):
        Rules("123")


#######
# get #
#######


def test_strategy_manager_get_returns_config():
    sm = StrategyManager()
    result = sm.get()
    assert isinstance(result, StratsDict)


##############
# pretty_get #
##############


def test_pretty_get_sequential_api(s1, s2):
    sm = StrategyManager()
    sm.apply(s1)
    sm.apply(s2)

    assert sm.pretty_get() == "[\n\tdummy_strategy(),\n\tdummy_strategy()\n]"


def test_pretty_get_dict_api(s1, s2, s3):
    sm = StrategyManager()
    sm.apply({"col_a": [s1, s3], "col_b": [s2]})

    pretty = sm.pretty_get()
    assert pretty == "{\n\t'col_a': (dummy_strategy(),\n\tdummy_strategy(),),\n\t'col_b': (dummy_strategy(),),\n}"

def test_pretty_get_rules_api(s1, s2, s3):
    sm = StrategyManager()
    sm.apply(Rules(on("col_a", s1), on("col_b", s2)))

    pretty = sm.pretty_get()
    assert pretty == "Rules(\n\ton('col_a', dummy_strategy()),\n\ton('col_b', dummy_strategy())\n)"


#########
# reset #
#########


def test_strategy_manager_reset_clears_strats(s1):
    sm = StrategyManager()

    assert sm.get() == {SEQUENTIAL_API_DEFAULT_KEY: []}
    assert sm.pretty_get() is None

    sm.apply(s1)
    sm.reset()

    assert sm.get() == {SEQUENTIAL_API_DEFAULT_KEY: []}
    assert sm.pretty_get() is None


###########################
# InvalidStrategyError #
###########################


def test_strategy_config_type_error_is_type_error():
    err = InvalidStrategyError("bad")
    assert isinstance(err, TypeError)
