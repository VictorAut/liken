"""Defines containers for strategies"""

from __future__ import annotations

from collections import UserDict
from copy import deepcopy
from typing import final

from liken._constants import INVALID_DICT_KEY_MSG
from liken._constants import INVALID_DICT_MEMBER_MSG
from liken._constants import INVALID_DICT_VALUE_MSG
from liken._constants import INVALID_FALLBACK_MSG
from liken._constants import INVALID_SEQUENCE_AFTER_DICT_MSG
from liken._constants import SEQUENTIAL_API_DEFAULT_KEY
from liken._constants import WARN_DICT_REPLACES_SEQUENCE_MSG
from liken._constants import WARN_RULES_REPLACES_RULES_MSG
from liken._dedupers import BaseDeduper
from liken._exceptions import InvalidStrategyError
from liken._exceptions import warn
from liken.rules import On
from liken.rules import Pipeline


# STRATS DICT CONFIG:


@final
class StratsDict(UserDict):
    """Container for combnations of strategies in the Sequential and Dict APIs

    For Sequential API all values (strategies) are added under a default key.

    For Dict API column label(s) (i.e. str or tuple) are the keys."""

    def __setitem__(self, key, value):
        if not isinstance(key, str | tuple):
            raise InvalidStrategyError(INVALID_DICT_KEY_MSG.format(type(key).__name__))
        if not isinstance(value, list | tuple | BaseDeduper):
            raise InvalidStrategyError(INVALID_DICT_VALUE_MSG.format(type(value).__name__))
        if not isinstance(value, BaseDeduper):
            for i, member in enumerate(value):
                if not isinstance(member, BaseDeduper):
                    raise InvalidStrategyError(INVALID_DICT_MEMBER_MSG.format(i, key, type(member).__name__))
        else:
            value = (value,)
        super().__setitem__(key, value)

    def __str__(self):
        rep = ""
        for k, values in self.items():
            krep = ""
            for v in values:
                krep += "\n\t\t" + str(v) + ","
            rep += f"\n\t'{k}': ({krep[:-1]},\n\t\t),"
        return "{" + rep + "\n}"


# STRATS MANAGER:


@final
class StrategyManager:
    """
    Manage and validate collection(s) of deduplication strategies.

    Supports addition of strategies as part of the three APIs:
    - Sequential
    - Dict
    - Pipeline

    For Sequential strategies, as instances of `BaseDeduper` are sequentially
    to an idential structure of the Dict API but under a single default
    dictionary key. Keys are columns names, and values are iterables of
    strategies.

    Raises:
        InvalidStrategyError for any misconfigured strategy
    """

    def __init__(self) -> None:
        self._strats: StratsDict | Pipeline = StratsDict({SEQUENTIAL_API_DEFAULT_KEY: []})
        self.has_applies: bool = False

    @property
    def is_sequential_applied(self) -> bool:
        """checks to see if stratgies are loaded under the default key"""
        if isinstance(self._strats, Pipeline):
            return False
        return set(self._strats) == {SEQUENTIAL_API_DEFAULT_KEY}
        

    def apply(self, strat: BaseDeduper | dict | StratsDict | Pipeline) -> None:
        """Loads a strategy into the manager

        This function currently handles all possible instances of strategy, and
        the implementation achieves this by writing to the strategy dictionary
        or overwriting the dictionary with `Pipeline`.

        If the input strat is `BaseDeduper` then "Sequential" API is in use. If
        dict (or StratsDict — even though this is not public) then it is the
        "Dict" API. Else "Pipeline" API is in use.

        Note also that as Pipeline contains On and combinations of On operated
        with & results in self mutation, need deep copy to allow for
        serialization to Spark workers."""

        # track that at least one apply made
        # if not, used by `Dedupe` to include an exact deduper by default
        self.has_applies = True

        if isinstance(strat, BaseDeduper):
            if not self.is_sequential_applied:
                raise InvalidStrategyError(INVALID_SEQUENCE_AFTER_DICT_MSG)
            self._strats[SEQUENTIAL_API_DEFAULT_KEY].append(strat)
            return

        if isinstance(strat, dict | StratsDict):
            if self._strats[SEQUENTIAL_API_DEFAULT_KEY]:
                warn(WARN_DICT_REPLACES_SEQUENCE_MSG)
            self._strats = StratsDict(strat)
            return

        if isinstance(strat, On):
            strat = Pipeline.step(strat)

        if isinstance(strat, Pipeline):
            if isinstance(self._strats, Pipeline):
                warn(WARN_RULES_REPLACES_RULES_MSG)
            # required for spark serialization
            self._strats = deepcopy(strat)  # type: ignore
            return

        raise InvalidStrategyError(INVALID_FALLBACK_MSG.format(type(strat).__name__))

    def get(self) -> StratsDict | Pipeline:
        return self._strats

    def pretty_get(self) -> None | str:
        """string representation of strats.

        Output string must be formatted approximately such that it can be used
        with .apply(), i.e. a string representation of one of:
            - BaseDeduper
            - StratsDict
            - Pipeline
        The seuqneital API with numerous additions of BaseStraegy means there
        is not good way to retried this such that is available to "apply". So,
        default to returning it as a list representation.
        """
        strats = self.get()

        if isinstance(strats, StratsDict):
            if self.is_sequential_applied:
                deduper: list = strats[SEQUENTIAL_API_DEFAULT_KEY]
                if not deduper:
                    return None
                return str(*deduper)
            return str(strats)
        return str(strats)

    def reset(self):
        """Reset strategy collection to empty defaultdict"""
        self._strats = StratsDict({SEQUENTIAL_API_DEFAULT_KEY: []})
