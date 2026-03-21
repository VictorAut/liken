"""Defines containers for strategies"""

from __future__ import annotations

from typing import Self
from typing import final

from liken._constants import INVALID_RULE_EMPTY_MSG
from liken._constants import INVALID_RULE_MEMBER_MSG
from liken._dedupers import BaseDeduper
from liken._dedupers import PredicateDedupers
from liken._exceptions import InvalidStrategyError
from liken._registry import registry
from liken._types import Columns
from liken._processors import Processor


# STRATS RULES CONFIG


class Pipeline(tuple):
    """Tuple-like container of strategies.

    Accepts single or multiple strategies where those strategies are passed
    with the ``on`` function.

    Args:
        *strategies: comma separated ``on`` strategies, unpacked.


    Example:
        A single strategy is passed:

            from liken import Dedupe, exact
            from liken.rules import Pipeline, on

            STRAT = Pipeline(on("address", exact()))

            lk = Dedupe(df)
            lk.apply(STRAT)

        Multiple strategies are passed:

            from liken import Dedupe, exact
            from liken.rules import Pipeline, on

            STRAT = Pipeline(
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
                raise InvalidStrategyError(
                    INVALID_RULE_MEMBER_MSG.format(i, type(item).__name__)
                )

        return super().__new__(cls, strategies)

    def __str__(self) -> str:
        inner = ",\n\t".join(str(s) for s in self)
        return f"lk.pipeline(\n\t{inner}\n)"


@final
class On:
    """Unit container for a single strategy in the Pipeline API"""

    # for IDE autocompletion only!
    # must be manually maintained
    # add a new dummy method here upon adding a new deduper.
    def exact(self, *args, **kwargs) -> On:
        return self.__getattr__("exact")(*args, **kwargs)

    def fuzzy(self, *args, **kwargs) -> On:
        return self.__getattr__("fuzzy")(*args, **kwargs)

    def tfidf(self, *args, **kwargs) -> On:
        return self.__getattr__("tfidf")(*args, **kwargs)

    def lsh(self, *args, **kwargs) -> On:
        return self.__getattr__("lsh")(*args, **kwargs)

    def jaccard(self, *args, **kwargs) -> On:
        return self.__getattr__("jaccard")(*args, **kwargs)

    def cosine(self, *args, **kwargs) -> On:
        return self.__getattr__("cosine")(*args, **kwargs)

    def isin(self, *args, **kwargs) -> On:
        return self.__getattr__("isin")(*args, **kwargs)

    def isna(self, *args, **kwargs) -> On:
        return self.__getattr__("isna")(*args, **kwargs)

    def str_startswith(self, *args, **kwargs) -> On:
        return self.__getattr__("str_startswith")(*args, **kwargs)

    def str_endswith(self, *args, **kwargs) -> On:
        return self.__getattr__("str_endswith")(*args, **kwargs)

    def str_contains(self, *args, **kwargs) -> On:
        return self.__getattr__("str_contains")(*args, **kwargs)

    def str_len(self, *args, **kwargs) -> On:
        return self.__getattr__("str_len")(*args, **kwargs)

    def __init__(self, columns: Columns, processors: Processor | list[Processor] = []):

        self._columns = columns
        self._dedupers: list[tuple[Columns, BaseDeduper]] = []
        self._processors: list[Processor] = (
            [processors] if isinstance(processors, Processor) else processors
        )

    def __and__(self, other: On) -> Self:
        """Combine multiple On instances with AND."""
        self._dedupers.extend(other._dedupers)
        return self

    def __invert__(self) -> On:
        """Propagate inverstion to the deduper Allows for following syntax:

        ~on("email").isna()

        Where the inversion get's propagated to act on isna().
        """

        columns, strat = self._dedupers[0]

        new_on = On(columns)
        new_on._dedupers = [(columns, ~strat)]
        return new_on

    def __getattr__(self, attr):
        """Makes deduper functions available as method calls to On.

        Functions are retrieved from registry. Includes any prior custom
        dedupers that have been registered.
        """

        # don't intercept Python internals
        if attr.startswith("__"):
            raise AttributeError(attr)

        func = registry.get(f"{attr}")

        def wrapper(*args, **kwargs):
            strat = func(*args, **kwargs)
            self._dedupers = [(self._columns, strat)]
            return self

        return wrapper

    @property
    def and_dedupers(self) -> list[tuple[Columns, BaseDeduper]]:
        """return strategies, sorted such that binary strategies are first.
        This is used for predication.
        """
        return sorted(
            self._dedupers, key=lambda x: not isinstance(x[1], PredicateDedupers)
        )

    @property
    def has_any_binary_strat(self) -> bool:
        """whether or not the Rule set of strategies has at least one
        Binary strategy.
        """
        return any([isinstance(x[1], PredicateDedupers) for x in self._dedupers])

    def __str__(self) -> str:
        """string representation

        Parses a single On or combinations of On operated with `&`
        """
        rep = ""
        for cs in self._dedupers:
            column: str = cs[0]
            deduper: str = str(cs[1])
            on = "on"
            if deduper.startswith("~"):
                deduper = deduper[1:]
                on = "~" + on
            rep += f"{on}('{column}').{deduper} & "
        return rep[:-3]


class Builder:

    def __init__(self, processors: Processor | list[Processor] = []):
        self._processors: list[Processor] = (
            [processors] if isinstance(processors, Processor) else processors
        )
        self._dedupers: list[On] = []

    def step(
        self,
        dedupers: On | list[On],
        /,
        *,
        processors: Processor | list[Processor] = [],
    ) -> Self:
        self._processors: list[Processor] = (
            [processors] if isinstance(processors, Processor) else processors
        )

        if isinstance(dedupers, On):
            self._dedupers.append([dedupers])
        else:
            self._dedupers.append(dedupers)

        return self
