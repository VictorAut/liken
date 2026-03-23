"""TODO"""

from __future__ import annotations

from typing import NamedTuple
from typing import Self
from typing import TypeAlias
from typing import final

from liken._dedupers import BaseDeduper
from liken._dedupers import PredicateDedupers
from liken._preprocessors import Preprocessor
from liken._registry import registry
from liken._types import Columns
from liken._validators import validate_preprocessor_arg


# TYPES:


class PipelineUnit(NamedTuple):
    columns: Columns
    deduper: BaseDeduper
    preprocessors: list[Preprocessor]


PipelineStep: TypeAlias = list[PipelineUnit]
PipelineCollection: TypeAlias = list[PipelineStep]

InputPreprocessor: TypeAlias = Preprocessor | list[Preprocessor]

# PUBLIC ON API:


def pipeline(preprocessors: InputPreprocessor = []) -> Pipeline:
    """TODO"""
    return Pipeline(preprocessors)


def on(columns: Columns, /, *, preprocessors: InputPreprocessor = []) -> On:
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
    return On(columns, preprocessors=preprocessors)


# ON:


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

    def __init__(self, columns: Columns, preprocessors: InputPreprocessor = []):
        self._columns: Columns = columns
        self._unit: PipelineUnit
        self._preprocessors: list[Preprocessor] = resolve_preprocessors(preprocessors)

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
            self._unit = PipelineUnit(self._columns, deduper, self._preprocessors)
            return self

        return wrapper

    def __invert__(self) -> On:
        """Propagate inverstion to the deduper. Allows for following syntax:

        ~on("email").isna()

        Where the inversion get's propagated to act on isna().
        """

        columns, deduper, preprocessors = self._unit

        new_on = On(columns)
        new_on._unit = PipelineUnit(columns, ~deduper, preprocessors)
        return new_on

    @property
    def unit(self) -> PipelineUnit:
        return self._unit

    def __str__(self) -> str:
        """string representation

        Parses a single On or combinations of On operated with `&`
        """
        on: str = "lk.rules.on"

        rep = ""
        columns, deduper, _ = self._unit
        deduper: str = str(deduper)
        if deduper.startswith("~"):
            deduper = deduper[1:]
            on = "~" + on
        rep += f"{on}('{columns}').{deduper}"
        return rep


# PIPELINE:


class Pipeline:
    """TODO: write docs"""

    def __init__(self, preprocessors: InputPreprocessor = []):
        self._preprocessors: list[Preprocessor] = resolve_preprocessors(preprocessors)
        self._ons: list[list[On]] = []
        self._steps: PipelineCollection = []

    def step(
        self,
        ons: On | list[On],
        /,
        *,
        preprocessors: InputPreprocessor = [],
    ) -> Self:
        """TODO: write docs"""
        preprocessors: list[Preprocessor] = resolve_preprocessors(preprocessors)
        if not preprocessors:
            preprocessors = self._preprocessors

        if isinstance(ons, On):
            ons: list[On] = [ons]
        self._ons.append(ons)

        step: PipelineStep = [on.unit for on in ons]

        # propagate preprocessor
        step = [s._replace(preprocessors=preprocessors) if not s.preprocessors else s for s in step]

        # predicates sorted to first for "rule predication"
        step: PipelineStep = sorted(step, key=lambda x: not isinstance(x[1], PredicateDedupers))

        self._steps.append(step)

        return self

    def __str__(self) -> str:
        pros = ""
        if preprocessors := self._preprocessors:
            pros = "preprocessors=" + f"{[str(p) for p in preprocessors]}"

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

        Private! not public API.
        """
        return any([isinstance(x[1], PredicateDedupers) for x in step])

    @property
    def steps(self):
        return self._steps


# HELPERS:


def resolve_preprocessors(
    preprocessors: InputPreprocessor,
) -> list[Preprocessor]:
    if isinstance(preprocessors, list):
        return [validate_preprocessor_arg(p) for p in preprocessors]
    # i.e. a single preprocessor
    return [validate_preprocessor_arg(preprocessors)]
