from collections import UserDict
import logging
from typing import final, Final

from dupegrouper.strats_library import BaseStrategy


# LOGGER:


logger = logging.getLogger(__name__)


# CONSTANTS:


DEFAULT_STRAT_KEY: Final[str] = "_default_"


# STRATS CONFIG:


@final
class StratsConfig(UserDict):
    def __setitem__(self, key, value):
        if not isinstance(key, str | tuple):
            raise StrategyConfigTypeError(
                f"Invalid type for dict key type: expected str or tuple, " f'got "{type(key).__name__}"'
            )
        if not isinstance(value, list | tuple | BaseStrategy):
            raise StrategyConfigTypeError(
                f"Invalid type for dict value: expected list, tuple or BaseStrategy, "
                f'got {value} "{type(value).__name__}"'
            )
        for i, member in enumerate(value):
            if not isinstance(member, BaseStrategy):
                raise StrategyConfigTypeError(
                    f"Invalid type for dict value member: at index {i} for key '{key}': "
                    f'expected "BaseStrategy", got "{type(member).__name__}"'
                )
        super().__setitem__(key, value)


@final
class StrategyManager:
    """
    Manage and validate collection(s) of deduplication strategies.

    Strategies are collected into a dictionary-like collection where keys are
    attribute names, and values are lists of strategies. Validation is provided
    upon addition allowing only the following stratgies types:
        - `BaseStrategy`
    A public property exposes stratgies upon successul addition and validation.
    A `StrategyConfigTypeError` is thrown, otherwise.
    """

    def __init__(self) -> None:
        self._strats = StratsConfig({DEFAULT_STRAT_KEY: []})

    def apply(self, strat: BaseStrategy | dict | StratsConfig) -> None:
        if isinstance(strat, BaseStrategy):
            if DEFAULT_STRAT_KEY not in self._strats:
                raise StrategyConfigTypeError(
                    "Cannot apply a BaseStrategy after a strategy mapping (dict) has been set. "
                    "Use either individual BaseStrategy instances or a dict of strategies, not both."
                )

            self._strats[DEFAULT_STRAT_KEY].append(strat)
            return

        if isinstance(strat, dict | StratsConfig):
            # when an inline strat has already been provided: warn as it will be replaced
            if self._strats[DEFAULT_STRAT_KEY]:
                # change this to use warnings library instead
                logger.warning(
                    "The strat manager had already been supplied with at least one in-line strat which now will be replaced"
                )

            self._strats = StratsConfig(strat)
            return

        raise StrategyConfigTypeError(f'Invalid strategy: Expected BaseStrategy or dict, got {type(strat).__name__}')

    def get(self) -> StratsConfig:
        return self._strats

    def pretty_get(self) -> tuple[str, ...] | dict[str, tuple[str, ...]]:
        """pretty get"""
        strategies = self.get()

        def _parse(values):
            return tuple(type(vx).__name__ for vx in values)

        if set(strategies) == {DEFAULT_STRAT_KEY}:
            return tuple(_parse(strategies[DEFAULT_STRAT_KEY]))
        return {key: _parse(values) for key, values in strategies.items()}

    def reset(self):
        """Reset strategy collection to empty defaultdict"""
        self._strats = StratsConfig({DEFAULT_STRAT_KEY: []})


# EXCEPTIONS:


@final
class StrategyConfigTypeError(TypeError):
    def __init__(self, msg):
        super().__init__(msg)
