from __future__ import annotations

from typing import Self
from dupegrouper.strats_library import BaseStrategy, exact, fuzzy, tfidf, lsh, str_contains


class On:
    def __init__(self, column: str, strat):
        self._column = column
        self._strat = strat
        self._strats = [(column, strat)]

    def __and__(self, other: On) -> Self:
        self._strats.append((other._column, other._strat))
        return self
    
    @property
    def and_strats(self) -> list[tuple[str, BaseStrategy]]:
        return self._strats


test = (
    On("email", exact()),
    On("address", fuzzy()) & On("address", lsh()) & On("address", str_contains(pattern="h")),
)

# for stage in test:
#     for col, strat in stage.and_strats:
#         print(col)
#         print(strat)
#         print("do something")
#     print("collecting......")
#     # break