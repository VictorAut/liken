"""Defines containers for strategies"""

from __future__ import annotations

import warnings
from collections import UserDict
from copy import deepcopy
from typing import Self
from typing import final

from liken._constants import INVALID_DICT_KEY_MSG
from liken._constants import INVALID_DICT_MEMBER_MSG
from liken._constants import INVALID_DICT_VALUE_MSG
from liken._constants import INVALID_FALLBACK_MSG
from liken._constants import INVALID_RULE_EMPTY_MSG
from liken._constants import INVALID_RULE_MEMBER_MSG
from liken._constants import INVALID_SEQUENCE_AFTER_DICT_MSG
from liken._constants import SEQUENTIAL_API_DEFAULT_KEY
from liken._constants import WARN_DICT_REPLACES_SEQUENCE_MSG
from liken._constants import WARN_RULES_REPLACES_RULES_MSG
from liken._strats_library import BaseStrategy
from liken._types import Columns
from liken._validators import validate_strat_arg


# STRATS DICT CONFIG:


@final
class StratsDict(UserDict):
    """Container for combnations of strategies in the Sequential and Dict APIs

    For Sequential API all values (strategies) are added under a default key.

    For Dict API column label(s) (i.e. str or tuple) are the keys."""

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


class Rules(tuple):
    """Tuple-like container of strategies.

    Accepts single or multiple strategies where those strategies are passed
    with the ``on`` function.

    Args:
        *strategies: comma separated ``on`` strategies, unpacked.


    Example:
        A single strategy is passed:

            from liken import Dedupe, exact
            from liken.rules import Rules, on

            STRAT = Rules(on("address", exact()))

            lk = Dedupe(df)
            lk.apply(STRAT)

        Multiple strategies are passed:

            from liken import Dedupe, exact
            from liken.rules import Rules, on

            STRAT = Rules(
                on('address', exact()),
                on('email', fuzzy(threshold=0.95)) & on('address', ~isna()),
            )

            lk = Dedupe(df)
            lk.apply(STRAT)
    """

    def __new__(cls, *strategies: On):
        if len(strategies) == 1 and isinstance(strategies[0], tuple):
            strategies = strategies[0]

        if not strategies:
            raise InvalidStrategyError(INVALID_RULE_EMPTY_MSG)

        for i, item in enumerate(strategies):
            if not isinstance(item, On):
                raise InvalidStrategyError(INVALID_RULE_MEMBER_MSG.format(i, type(item).__name__))

        return super().__new__(cls, strategies)


@final
class On:
    """Unit container for a single strategy in the Rules API"""

    def __init__(self, columns: Columns, strat: BaseStrategy):
        self._columns = columns
        self._strat = validate_strat_arg(strat)
        self._strats: list[tuple[Columns, BaseStrategy]] = [(columns, strat)]

    def __and__(self, other: On) -> Self:
        """Overloads `&` operator

        Mutates first instance of On in chain of `&` operates On instances.
        Collectes the combinations of strategies into a single iterable for
        that step. The executor will then apply all these combined strategies
        and select union find components that satisfy all the combinations.

        Usage:
            On("address", exact) & On("email", isna())

        Returns:
            Self, specifically Self of the first On.
        """
        self._strats.append((other._columns, other._strat))
        return self

    @property
    def and_strats(self) -> list[tuple[Columns, BaseStrategy]]:
        return self._strats

    def __str__(self):
        """string representation

        Parses a single On or combinations of On operated with `&`
        """
        rep = ""
        for cs in self._strats:
            rep += f"on('{cs[0]}', {str(cs[1])}) & "
        return rep[:-3]


# STRATS MANAGER:


@final
class StrategyManager:
    """
    Manage and validate collection(s) of deduplication strategies.

    Supports addition of strategies as part of the three APIs:
    - Sequential
    - Dict
    - Rules

    For Sequential strategies, as instances of `BaseStrategy` are sequentially
    to an idential structure of the Dict API but under a single default
    dictionary key. Keys are columns names, and values are iterables of
    strategies.

    Raises:
        InvalidStrategyError for any misconfigured strategy
    """

    def __init__(self) -> None:
        self._strats: StratsDict | Rules = StratsDict({SEQUENTIAL_API_DEFAULT_KEY: []})
        self.has_applies: bool = False

    @property
    def is_sequential_applied(self) -> bool:
        """checks to see if stratgies are loaded under the default key"""
        return set(self._strats) == {SEQUENTIAL_API_DEFAULT_KEY}

    def apply(self, strat: BaseStrategy | dict | StratsDict | Rules) -> None:
        """Loads a strategy into the manager

        This function currently handles all possible instances of strategy, and
        the implementation achieves this by writing to the strategy dictionary
        or overwriting the dictionary with `Rules`.

        If the input strat is `BaseStrategy` then "Sequential" API is in use. If
        dict (or StratsDict — even though this is not public) then it is the
        "Dict" API. Else "Rules" API is in use.

        Note also that as Rules contains On and combinations of On operated
        with & results in self mutation, need deep copy to allow for
        serialization to Spark workers."""

        # track that at least one apply made
        # if not, used by `Dedupe` to include an exact deduper by default
        self.has_applies = True

        if isinstance(strat, BaseStrategy):
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

    def pretty_get(self) -> None | str:
        """string representation of strats.

        Output string must be formatted approximately such that it can be used
        with .apply(), i.e. a string representation of one of:
            - BaseStrategy
            - StratsDict
            - Rules
        The seuqneital API with numerous additions of BaseStraegy means there
        is not good way to retried this such that is available to "apply". So,
        default to returning it as a list representation.
        """
        strats = self.get()

        if isinstance(strats, StratsDict):
            # added as BaseStrategy (Sequential API)
            if self.is_sequential_applied:

                # short-circuit; nothing yet applied.
                if not strats[SEQUENTIAL_API_DEFAULT_KEY]:
                    return None

                rep = ""
                for strat in strats[SEQUENTIAL_API_DEFAULT_KEY]:
                    rep += str(strat) + ",\n\t"
                return f"[\n\t{rep[:-3]}\n]"

            # Normal dict API
            rep = ""
            for k, values in strats.items():
                krep = ""
                for v in values:
                    krep += str(v) + ",\n\t"
                rep += f"\n\t'{k}': ({krep[:-3]},),"
            return "{" + rep + "\n}"

        # Rules API
        if isinstance(strats, tuple):
            rep = ""
            for ons in strats:
                rep += str(ons) + ",\n\t"
            return f"Rules(\n\t{rep[:-3]}\n)"

    def reset(self):
        """Reset strategy collection to empty defaultdict"""
        self._strats = StratsDict({SEQUENTIAL_API_DEFAULT_KEY: []})


# EXCEPTIONS:


@final
class InvalidStrategyError(TypeError):
    def __init__(self, msg):
        super().__init__(msg)


def warn(msg: str) -> None:
    return warnings.warn(msg, category=UserWarning)


# PUBLIC ON API:


def on(columns: Columns, strat: BaseStrategy, /) -> None:
    """Unit container for a single strategy in the Rules API.

    Operates a "strat" on a "columns". Is provided as comma separated members to
    `Rules`. Allows for "and" chaining via the `&` operator to logically
    compose strategy "rules".

    The `&` ("and") operator is the only supplier logical combination operator
    supplier, as the equivalent to "or" is achieved by comma separating `on`
    calls inside `Rules`. The results of `&` are interepreted as boolean and
    whereby the left-hand deduplication strategy must agree with the right-hand
    strategy for any given pairwise combination.

    Args:
        columns: the label(s) of a column or columns.
        strat: the strategy to apply.

    Returns:
        None

    Example:
        single ``on`` strategy:

            from liken import Dedupe, exact, fuzzy
            from liken.rules import Rules, on, isna, str_endswith

            on("address", exact())

        Strategies combined with ``&``:

            on("email", fuzzy(threshold=0.95)) & on("email", str_endswith("UK"))

        Strategies can be combined with ``&`` for **different** columns:

            on("email", fuzzy(threshold=0.95)) & on("address", ~isna())

        The above can be read as "deduplicate email only when the address field
        is not null":

            >>> df # Before
            +------+-----------+---------------------+
            | id   |  address  |        email        |
            +------+-----------+---------------------+
            |  1   |  london   |  foobar@gmail.com   |
            |  2   |   paris   |  Foobar@gmail.com   |
            |  3   |   null    |  fooBar@gmail.com   |
            +------+-----------+---------------------+

            >>> df # After
            +------+-----------+---------------------+--------------+
            | id   |  address  |        email        | canonical_id |
            +------+-----------+---------------------+--------------+
            |  1   |  london   |  foobar@gmail.com   |       1      |
            |  2   |   paris   |  Foobar@gmail.com   |       1      |
            |  3   |   null    |  fooBar@gmail.com   |       3      |
            +------+-----------+---------------------+--------------+

        Where the first two rows are now linked via the same canonical_id.
    """
    return On(columns, strat)
