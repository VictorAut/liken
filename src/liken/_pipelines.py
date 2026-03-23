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
    """Convenience function for `Pipeline` collection."""
    return Pipeline(preprocessors)


def on(columns: Columns, /, *, preprocessors: InputPreprocessor = []) -> On:
    """Convenience function for calling `On` with a deduper."""
    return On(columns, preprocessors=preprocessors)


# ON:


@final
class On:
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

            lk.pipeline().step(on("address").exact())

        Dedupers are combined as "and" statements when passed as a members
        of a list in a step:

            lk.pipeline().step(
                [
                    on("email").fuzzy(threshold=0.95),
                    on("email").str_endswith("UK"),
                ]
            )

        Dedupers can be combined as "and" statement in a step for
        **different** columns:

            lk.pipeline().step(
                [
                    on("email").fuzzy(threshold=0.95),
                    ~on("address").isna(),
                ]
            )

        Pipeline preprocessors will be applied to all dedupers:
                
            pipeline = (
                lk.pipeline(preprocessors=[lk.preprocessors.lower()])
                .step(lk.on("email").fuzzy()) # Preprocessor applies here
                .step(lk.on("address").tfidf()) # And here
            )

        But, a global pipeline preprocessor will not override an explicit
        deduper's preprocessor. Similarily, a step's preprocessor will not
        override the deduper's:
            
            pipeline = (
                lk.pipeline(preprocessors=[lk.preprocessors.ascii_fold()])
                .step(
                    [
                        lk.on("email").fuzzy(),  # preprocessed by step's preprocessor, `alnum`.
                        ~lk.on(
                            "address",
                            preprocessors=[lk.preprocessors.lower()], 
                        ).isna(), # uses it's own preprocessor, `lower`.
                    ],
                    preprocessors=[lk.preprocessors.alnum()], # defines the step's preprocessor
                )
                .step(lk.on("address").tfidf()) # defaults to the pipeline's preprocessor, `ascii_fold`.
            )
    """

    # for IDE autocompletion only!
    # must be manually maintained
    # add a new dummy method here upon adding a new deduper.

    def exact(self, *args, **kwargs) -> On:
        """Method wrapper of `lk.exact` function.

        Usage is identical to the function but chained to an instance of `On`.
        See [`lk.exact`](../reference/liken.md#liken.exact) reference for
        complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(on("col").exact())

        """
        return self.__getattr__("exact")(*args, **kwargs)

    def fuzzy(self, *args, **kwargs) -> On:
        """Method wrapper of `lk.fuzzy` function.

        Usage is identical to the function but chained to an instance of `On`.
        See [`lk.fuzzy`](../reference/liken.md#liken.fuzzy) reference for
        complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(on("col").fuzzy())

        """
        return self.__getattr__("fuzzy")(*args, **kwargs)

    def tfidf(self, *args, **kwargs) -> On:
        """Method wrapper of `lk.tfidf` function.

        Usage is identical to the function but chained to an instance of `On`.
        See [`lk.tfidf`](../reference/liken.md#liken.tfidf) reference for
        complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(on("col").tfidf())

        """
        return self.__getattr__("tfidf")(*args, **kwargs)

    def lsh(self, *args, **kwargs) -> On:
        """Method wrapper of `lk.lsh` function.

        Usage is identical to the function but chained to an instance of `On`.
        See [`lk.lsh`](../reference/liken.md#liken.lsh) reference for
        complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(on("col").lsh())

        """
        return self.__getattr__("lsh")(*args, **kwargs)

    def jaccard(self, *args, **kwargs) -> On:
        """Method wrapper of `lk.jaccard` function.

        Usage is identical to the function but chained to an instance of `On`.
        See [`lk.jaccard`](../reference/liken.md#liken.jaccard) reference for
        complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(on("col").jaccard())

        """
        return self.__getattr__("jaccard")(*args, **kwargs)

    def cosine(self, *args, **kwargs) -> On:
        """Method wrapper of `lk.cosine` function.

        Usage is identical to the function but chained to an instance of `On`.
        See [`lk.cosine`](../reference/liken.md#liken.cosine) reference for
        complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(on("col").cosine())

        """
        return self.__getattr__("cosine")(*args, **kwargs)

    def isin(self, *args, **kwargs) -> On:
        """Method wrapper of `lk.isin` function.

        Usage is identical to the function but chained to an instance of `On`.
        See [`lk.isin`](../reference/liken.md#liken.isin) reference for
        complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(on("col").isin())

        """
        return self.__getattr__("isin")(*args, **kwargs)

    def isna(self, *args, **kwargs) -> On:
        """Method wrapper of `lk.isna` function.

        Usage is identical to the function but chained to an instance of `On`.
        See [`lk.isna`](../reference/liken.md#liken.isna) reference for
        complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(on("col").isna())

        """
        return self.__getattr__("isna")(*args, **kwargs)

    def str_startswith(self, *args, **kwargs) -> On:
        """Method wrapper of `lk.str_startswith` function.

        Usage is identical to the function but chained to an instance of `On`.
        See [`lk.str_startswith`](../reference/liken.md#liken.str_startswith) reference for
        complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(on("col").str_startswith())

        """
        return self.__getattr__("str_startswith")(*args, **kwargs)

    def str_endswith(self, *args, **kwargs) -> On:
        """Method wrapper of `lk.str_endswith` function.

        Usage is identical to the function but chained to an instance of `On`.
        See [`lk.str_endswith`](../reference/liken.md#liken.str_endswith) reference for
        complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(on("col").str_endswith())

        """
        return self.__getattr__("str_endswith")(*args, **kwargs)

    def str_contains(self, *args, **kwargs) -> On:
        """Method wrapper of `lk.str_contains` function.

        Usage is identical to the function but chained to an instance of `On`.
        See [`lk.str_contains`](../reference/liken.md#liken.str_contains) reference for
        complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(on("col").str_contains())

        """
        return self.__getattr__("str_contains")(*args, **kwargs)

    def str_len(self, *args, **kwargs) -> On:
        """Method wrapper of `lk.str_len` function.

        Usage is identical to the function but chained to an instance of `On`.
        See [`lk.str_len`](../reference/liken.md#liken.str_len) reference for
        complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(on("col").str_len())

        """
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
        on: str = "lk.on"

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
    """Builds deduplication pipeline.
    
    Defines deduplication steps as chainable calls. Accepts preprocessors.
    Preprocessors are propagate to each step and on each column, unless
    explicit overriden in those stages.
    
    Args:
        preprocessors: a preprocessor, or list of preprocessors to apply to the
            pipeline
    
    Returns:
        None

    Raises:
        TypeError: if passed preprocessor not a member of `liken.preprocessors`
    """

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
        """Define a deduplication pipeline step
        
        Accepts a single deduper or a list of dedupers. A list of dedupers that
        has more than one member are composed as "and rules" meaning that the
        deduplication of both (or more) dedupers are considered for 
        canonicalisation.
        
        Args:
            ons: `On` deduper, or list of the same.
            preprocessors: a preprocessor, or list of preprocessors to appy the
                whole step (i.e. to all dedupers if more than one deduper).
        
        Returns:
            Self
        
        Example:
            A single deduper is added:
                
                pipeline = lk.pipeline().step(lk.on("email").exact())

            Multiple steps can be chained:
                
                pipeline = (
                    lk.pipeline()
                    .step(lk.on("email").fuzzy())
                    .step(lk.on("address").tfidf())
                )

            More than on deduper can be added to form composable rules within
            a step:
                
                pipeline = (
                    lk.pipeline()
                    .step(
                        [
                            lk.on("email").fuzzy(),
                            ~lk.on("address").isna(),
                        ]
                    )
                    .step(lk.on("address").tfidf())
                )

            Pipeline preprocessors will be applied to all steps:
                
                pipeline = (
                    lk.pipeline(preprocessors=[lk.preprocessors.lower()])
                    .step(lk.on("email").fuzzy()) # Preprocessor applies here
                    .step(lk.on("address").tfidf()) # And here
                )

            But, a global pipeline preprocessor will not override an explicit
            step's preprocessor
                
                pipeline = (
                    lk.pipeline(preprocessors=[lk.preprocessors.lower()])
                    .step(
                        lk.on("email").fuzzy(), preprocessors=[lk.preprocessors.alnum()]
                    )  # only `alnum` preprocessed
                    .step(lk.on("address").tfidf())  # this one still preprocessed with `lower`
                )
        """
        preprocessors: list[Preprocessor] = resolve_preprocessors(preprocessors)
        if not preprocessors:
            preprocessors = self._preprocessors

        if isinstance(ons, On):
            ons: list[On] = [ons]
        self._ons.append(ons)

        step: PipelineStep = [on.unit for on in ons]

        # propagate preprocessor
        step = [
            s._replace(preprocessors=preprocessors) if not s.preprocessors else s
            for s in step
        ]

        # predicates sorted to first for "rule predication"
        step: PipelineStep = sorted(
            step, key=lambda x: not isinstance(x[1], PredicateDedupers)
        )

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
