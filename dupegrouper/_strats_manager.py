from __future__ import annotations

import warnings
from collections import UserDict
from copy import deepcopy
from typing import Final, Self, final

from dupegrouper._strats_library import BaseStrategy
from dupegrouper._validators import validate_strat_arg


# CONSTANTS:


DEFAULT_STRAT_KEY: Final[str] = "_default_"

INVALID_DICT_KEY_MSG: Final[str] = "Invalid type for dict key type: expected str or tuple, got '{}'"
INVALID_DICT_VALUE_MSG: Final[str] = "Invalid type for dict value: expected list, tuple or 'BaseStrategy', got '{}'"
INVALID_DICT_MEMBER_MSG: Final[str] = (
    "Invalid type for dict value member: at index {} for key '{}': 'expected 'BaseStrategy', got '{}'"
)
INVALID_SEQUENCE_AFTER_DICT_MSG: Final[str] = (
    "Cannot apply a 'BaseStrategy' after a strategy mapping (dict) has been set. "
    "Use either individual 'BaseStrategy' instances or a dict of strategies, not both."
)
INVALID_RULE_EMPTY_MSG: Final[str] = "Rules cannot be empty"
INVALID_RULE_MEMBER_MSG: Final[str] = "Invalid Rules element at index {} is not an instance of On, got '{}'"
INVALID_FALLBACK_MSG: Final[str] = "Invalid strategy: Expected a 'BaseStrategy', a dict or 'Rules', got '{}'"

WARN_DICT_REPLACES_SEQUENCE_MSG: Final[str] = "Replacing previously added sequence strategy with a dict strategy"
WARN_RULES_REPLACES_RULES_MSG: Final[str] = "Replacing previously added 'Rules' strategy with a new 'Rules' strategy"


# STRATS DICT CONFIG:


@final
class StratsDict(UserDict):
    def __setitem__(self, key, value):
        if not isinstance(key, str | tuple):
            raise InvalidStrategyError(INVALID_DICT_KEY_MSG.format(type(key).__name__))
        if not isinstance(value, list | tuple | BaseStrategy):
            raise InvalidStrategyError(INVALID_DICT_VALUE_MSG.format(type(value).__name__))
        if not isinstance(value, BaseStrategy):
            for i, member in enumerate(value):
                if not isinstance(member, BaseStrategy):
                    raise InvalidStrategyError(INVALID_DICT_MEMBER_MSG.format(i, key, type(member).__name__))
        else:
            value = (value,)
        super().__setitem__(key, value)


# STRATS RULES CONFIG


@final
class Rules(tuple):
    def __new__(cls, *items):
        if len(items) == 1 and isinstance(items[0], tuple):
            items = items[0]

        if not items:
            raise InvalidStrategyError(INVALID_RULE_EMPTY_MSG)

        for i, item in enumerate(items):
            if not isinstance(item, On):
                raise InvalidStrategyError(INVALID_RULE_MEMBER_MSG.format(i, type(item).__name__))

        return super().__new__(cls, items)


@final
class On:
    def __init__(self, column: str, strat: BaseStrategy):
        self._column = column
        self._strat = validate_strat_arg(strat)
        self._strats: list[tuple[str, BaseStrategy]] = [(column, strat)]

    def __and__(self, other: On) -> Self:
        self._strats.append((other._column, other._strat))
        return self

    @property
    def and_strats(self) -> list[tuple[str, BaseStrategy]]:
        return self._strats


def on(column: str, strat: BaseStrategy, /):
    """here's how you define a single strategy in a Rule"""
    return On(column, strat)


# STRATS MANAGER:


@final
class StrategyManager:
    """
    Manage and validate collection(s) of deduplication strategies.

    Strategies are collected into a dictionary-like collection where keys are
    attribute names, and values are lists of strategies. Validation is provided
    upon addition allowing only the following stratgies types:
        - `BaseStrategy`
    A public property exposes stratgies upon successul addition and validation.
    A `InvalidStrategyError` is thrown, otherwise.
    """

    def __init__(self) -> None:
        self._strats = StratsDict({DEFAULT_STRAT_KEY: []})

    def apply(self, strat: BaseStrategy | dict | StratsDict | Rules) -> None:

        if isinstance(strat, BaseStrategy):
            if DEFAULT_STRAT_KEY not in self._strats:
                raise InvalidStrategyError(INVALID_SEQUENCE_AFTER_DICT_MSG)
            self._strats[DEFAULT_STRAT_KEY].append(strat)
            return

        if isinstance(strat, dict | StratsDict):
            if self._strats[DEFAULT_STRAT_KEY]:
                warn(WARN_DICT_REPLACES_SEQUENCE_MSG)
            self._strats = StratsDict(strat)
            return

        if isinstance(strat, On):
            strat = (strat,)

        if isinstance(strat, Rules | tuple):
            if isinstance(self._strats, Rules):
                warn(WARN_RULES_REPLACES_RULES_MSG)

            # Contents of Rules is mutable!
            # `On` operated on with `&` results in modified `On`
            # Of which only the first one is preserved
            # To guarantee repeated use of the base class, require deepcopy
            self._strats = Rules(deepcopy(strat))
            return

        raise InvalidStrategyError(INVALID_FALLBACK_MSG.format(type(strat).__name__))

    def get(self) -> StratsDict | Rules:
        return self._strats

    def pretty_get(self) -> tuple[str, ...] | dict[str, tuple[str, ...]]:
        """pretty get"""
        strats = self.get()

        def _parse(values):
            return tuple(str(v) for v in values)

        if set(strats) == {DEFAULT_STRAT_KEY}:
            return tuple(_parse(strats[DEFAULT_STRAT_KEY]))
        return {k: _parse(v) for k, v in strats.items()}

    def reset(self):
        """Reset strategy collection to empty defaultdict"""
        self._strats = StratsDict({DEFAULT_STRAT_KEY: []})


# EXCEPTIONS:


@final
class InvalidStrategyError(TypeError):
    def __init__(self, msg):
        super().__init__(msg)


def warn(msg: str) -> Warning:
    return warnings.warn(msg, category=UserWarning)
