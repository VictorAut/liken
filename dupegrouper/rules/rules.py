from .._strats_library import BaseStrategy, StrContains, StrEndsWith, StrStartsWith, IsNA
from .._strats_manager import Rules as _Rules
from .._strats_manager import on as _on


class Rules(_Rules):
    """TODO"""

    pass


def on(column: str, strat: BaseStrategy, /):
    """TODO"""
    return _on(column, strat)


# STRATEGIES:


def isna():
    """TODO"""
    return IsNA()

def str_startswith(pattern: str, case: bool = True) -> BaseStrategy:
    """TODO"""
    return StrStartsWith(pattern=pattern, case=case)


def str_endswith(pattern: str, case: bool = True) -> BaseStrategy:
    """TODO"""
    return StrEndsWith(pattern=pattern, case=case)


def str_contains(
    pattern: str,
    case: bool = True,
    regex: bool = False,
) -> BaseStrategy:
    """TODO"""
    return StrContains(pattern=pattern, case=case, regex=regex)
