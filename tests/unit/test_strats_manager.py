
##############################################
#  TEST StrategyManager + StrategyConfigTypeError #
##############################################


DEFAULT_ERROR_MSG = "Input is not valid"
CLASS_ERROR_MSG = "Input class is not valid: must be an instance of `BaseStrategy`"
TUPLE_ERROR_MSG = "Input tuple is not valid: must be a length 2 [callable, dict]"
DICT_ERROR_MSG = "Input dict is not valid: items must be a list of `BaseStrategy` or tuples"


@pytest.mark.parametrize(
    "strategy, expected_to_pass, base_msg",
    [
        # correct base inputs
        (Mock(spec=BaseStrategy), True, None),
        (
            {
                "address": [
                    Mock(spec=BaseStrategy),
                ],
                "email": [
                    Mock(spec=BaseStrategy),
                    Mock(spec=BaseStrategy),
                ],
            },
            True,
            None,
        ),
        # incorrect inputs
        (DummyClass, False, CLASS_ERROR_MSG),
        (lambda x: x, False, DEFAULT_ERROR_MSG),
        ((lambda x: x, [1, 2, 3]), False, TUPLE_ERROR_MSG),
        (("foo",), False, TUPLE_ERROR_MSG),
        (["bar", "baz"], False, DEFAULT_ERROR_MSG),
        ("foobar", False, DEFAULT_ERROR_MSG),
        (
            {
                "address": [DummyClass()],
                "email": [
                    "random string",
                    ("tuple too short",),
                ],
            },
            False,
            DICT_ERROR_MSG,
        ),
    ],
    ids=[
        "valid canonicalize class",
        "valid dict",
        "invalid class",
        "invalid callable not in tuple",
        "invalid callable positional args",
        "invalid tuple",
        "invalid list",
        "invalid str",
        "invalid dict",
    ],
)
def test__strategy_manager_validate_addition_strategy(strategy, expected_to_pass, base_msg):
    """validates that the input 'strtagey' is legit, against `StrategyConfigTypeError`"""
    manager = StrategyManager()
    if expected_to_pass:
        if isinstance(strategy, dict):
            for k, value in strategy.items():
                for v in value:
                    manager.add(k, v)
                    assert (k in manager.get()) is expected_to_pass
        else:
            manager.add(DEFAULT_STRAT_KEY, strategy)
            assert (DEFAULT_STRAT_KEY in manager.get()) is expected_to_pass
    else:
        with pytest.raises(StrategyConfigTypeError) as e:
            manager.add(DEFAULT_STRAT_KEY, strategy)
            assert base_msg in str(e)


def test__strategy_manager_reset():
    manager = StrategyManager()
    strategy = Mock(spec=BaseStrategy)
    manager.add("name", strategy)
    manager.reset()
    assert manager.get() == {}