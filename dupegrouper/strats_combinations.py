from __future__ import annotations

from typing import Any, Self
from dupegrouper.strats_library import BaseStrategy, exact, fuzzy, lsh, str_contains


class On:
    def __init__(self, column: str, strat: BaseStrategy):
        self._column = column
        self._strat = _validate_strat_arg(strat)
        self._strats: list[tuple[str, BaseStrategy]] = [(column, strat)]

    def __and__(self, other: On) -> Self:
        self._strats.append((other._column, other._strat))
        return self
    
    @property
    def and_strats(self) -> list[tuple[str, BaseStrategy]]:
        return self._strats


def _validate_strat_arg(strat: Any):
    if not isinstance(strat, BaseStrategy):
        raise TypeError(f"Invalid arg: strat must be instance of BaseStrategy, got {type(strat).__name__}")
    return strat
