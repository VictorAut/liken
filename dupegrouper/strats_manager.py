from collections import UserDict
from typing import final

from dupegrouper.constants import DEFAULT_STRAT_KEY
from dupegrouper.strats_library import BaseStrategy


# STRATS CONFIG:

@final
class StratsConfig(UserDict):
    def __setitem__(self, key, value):
        if not isinstance(key, str | tuple):
            raise StrategyConfigTypeError(
                f'Invalid type for strat dict key: '
                f'expected "str" or "tuple", got "{type(key).__name__}"'
            )
        if not isinstance(value, list | tuple | BaseStrategy):
            raise StrategyConfigTypeError(
                f'Invalid type for strat dict value: '
                f'expected "list", "tuple" or "BaseStrategy", got {value} "{type(value).__name__}"'
            )
        for i, member in enumerate(value):
            if not isinstance(member, BaseStrategy):
                    raise StrategyConfigTypeError(
                        f'Invalid type for strat dict value member: '
                        f'position {i} of value for key "{key}". '
                        f'Expected "BaseStrategy", got "{type(member).__name__}"'
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

    def apply(self, strategy: BaseStrategy | dict | StratsConfig) -> None:
        if isinstance(strategy, BaseStrategy):
            self._strats[DEFAULT_STRAT_KEY].append(strategy) 
            return
        if isinstance(strategy, dict | StratsConfig):
            self._strats = StratsConfig(strategy)
            return
        raise StrategyConfigTypeError(f'Invalid strategy: Expected "BaseStrategy" or "dict", got {type(strategy)}')

    def get(self) -> StratsConfig:
        return self._strats

    def pretty_get(self)-> None | tuple[str, ...] | dict[str, tuple[str, ...]]:
        """pretty get"""
        strategies = self.get()
        if not strategies:
            return None

        def _parse(values):
            return tuple(type(vx).__name__ for vx in values)

        if set(strategies) == {DEFAULT_STRAT_KEY}:
            return tuple(_parse(strategies[DEFAULT_STRAT_KEY]))
        return {key: _parse(values) for key, values in strategies.items()}

    def reset(self):
        """Reset strategy collection to empty defaultdict"""
        self._strats.clear()



# EXCEPTIONS:


@final
class StrategyConfigTypeError(TypeError):
    def __init__(self, msg):
        super().__init__(msg)