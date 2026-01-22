from __future__ import annotations

from typing import Self

from dupegrouper.strats_library import BaseStrategy
from dupegrouper.validators import validate_strat_arg


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


# PUBLIC API:

def on(column: str, strat: BaseStrategy, /):
    return On(column, strat)