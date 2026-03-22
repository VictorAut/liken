"""TODO"""

from __future__ import annotations

from typing import Self
from typing import final


from liken._dedupers import BaseDeduper
from liken._dedupers import PredicateDedupers
from liken._processors import Processor
from liken._registry import registry
from liken._types import Columns


@final
class On:
    """Unit container for a single deduper in the Pipeline API"""

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
        self._columns: Columns = columns
        self._dedupers: tuple[Columns, BaseDeduper] = ()
        self._processors: list[Processor] = [processors] if isinstance(processors, Processor) else processors

    def __invert__(self) -> On:
        """Propagate inverstion to the deduper Allows for following syntax:

        ~on("email").isna()

        Where the inversion get's propagated to act on isna().
        """

        columns, deduper = self._dedupers

        new_on = On(columns)
        new_on._dedupers = (columns, ~deduper)
        return new_on

    def __getattr__(self, attr):
        """Make deduper functions available as method calls to On.

        Functions are retrieved from registry. Includes any prior custom
        dedupers that have been registered.
        """

        # No intercept: Python internals
        if attr.startswith("__"):
            raise AttributeError(attr)

        func = registry.get(f"{attr}")

        def wrapper(*args, **kwargs):
            deduper = func(*args, **kwargs)
            self._dedupers = (self._columns, deduper)
            return self

        return wrapper

    @property
    def dedupers(self) -> tuple[Columns, BaseDeduper]:
        return self._dedupers

    def __str__(self) -> str:
        """string representation

        Parses a single On or combinations of On operated with `&`
        """
        on: str = "lk.rules.on"

        rep = ""
        columns, deduper = self._dedupers
        deduper: str = str(deduper)
        if deduper.startswith("~"):
            deduper = deduper[1:]
            on = "~" + on
        rep += f"{on}('{columns}').{deduper}"
        return rep


class Pipeline:
    """TODO: write docs"""
    def __init__(self, processors: Processor | list[Processor] = []):
        self._processors: list[Processor] = [processors] if isinstance(processors, Processor) else processors
        self._ons: list[list[On]] = []
        self._dedupers: list[list[tuple[Columns, BaseDeduper]]] = []

    def step(
        self,
        ons: On | list[On],
        /,
        *,
        processors: Processor | list[Processor] = [],
    ) -> Self:
        """TODO: write docs"""
        processors: list[Processor] = [processors] if isinstance(processors, Processor) else processors

        if isinstance(ons, On):
            ons: list[On] = [ons]
        self._ons.append(ons)

        dedupers = [on.dedupers for on in ons]
        # predicates sorted to first for "rule predication"
        dedupers = sorted(dedupers, key=lambda x: not isinstance(x[1], PredicateDedupers))

        self._dedupers.append(dedupers)

        return self

    def __str__(self) -> str:
        pros = ""
        if processors := self._processors:
            pros = "processors=" + f"{[str(p) for p in processors]}"

        inner = ""
        for step in self._ons:
            inner += "\n\t\t" + ".step(["
            for on in step:
                inner += "\n\t\t\t" + str(on) + ","
            inner += "\n\t\t" + "])"
        return f"(\n\tlk.rules.builder({pros}){inner}\n)"

    @staticmethod
    def _has_any_predicate(step: list[tuple[Columns, BaseDeduper]]) -> bool:
        """Retrieve whether or not the pipeline step has at least one predicate
        deduper.

        Private; not public API
        """
        return any([isinstance(x[1], PredicateDedupers) for x in step])

    @property
    def dedupers(self):
        return self._dedupers





# PUBLIC ON API:


def pipeline(processors: Processor | list[Processor] = []) -> Pipeline:
    """TODO"""
    return Pipeline(processors)


def on(columns: Columns, /) -> On:
    """Unit collection for a single deduper in the Pipeline API.

    Operates deduplication on a column. Passed as a step in a deduplication
    pipeline. Allows for usage of dedupers as method calls instead of functions.

    Args:
        columns: the label(s) of a column or columns.

    Returns:
        None

    Example:
        single `on` in a pipeline step:

            import liken as lk

            lk.rules.pipeline().step(on("address").exact())

        Dedupers are combined as "and" statements when passed as a members
        of a list in a step:

            lk.rules.pipeline().step(
                [
                    on("email").fuzzy(threshold=0.95),
                    on("email").str_endswith("UK"),
                ]
            )

        Dedupers can be combined as "and" statement in a step for
        **different** columns:

            lk.rules.pipeline().step(
                [
                    on("email").fuzzy(threshold=0.95),
                    ~on("address").isna(),
                ]
            )

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
    return On(columns)
