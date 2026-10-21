"""Defines the Pipeline collection: a `Pipeline` of `PipelineStep`s of `Col`s"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import Self
from typing import TypeAlias
from typing import cast
from typing import final

from liken.constants import INVALID_EMPTY_PIPELINE_STEP_MSG
from liken.constants import INVALID_NO_DEDUPER_CHAINED_MSG
from liken.constants import WARN_COL_RECHAINED_MSG
from liken.constants import WARN_COL_REUSED_MSG
from liken.core.deduper import BaseDeduper
from liken.core.deduper import PredicateDeduper
from liken.core.registries import dedupers_registry
from liken.exceptions import InvalidArgumentValueError
from liken.exceptions import InvalidDeduperError
from liken.exceptions import warn
from liken.preprocessors import Preprocessor
from liken.validations import require_type


if TYPE_CHECKING:
    from collections.abc import Callable

    from liken.types import Columns


# TYPES:


class PipelineUnit(NamedTuple):
    columns: Columns
    deduper: BaseDeduper
    preprocessors: list[Preprocessor]


PipelineStep: TypeAlias = list[PipelineUnit]
PipelineCollection: TypeAlias = list[PipelineStep]

InputPreprocessor: TypeAlias = Preprocessor | list[Preprocessor] | tuple[Preprocessor, ...]


class _Inherit:
    """Sentinel type for an omitted `preprocessors` argument."""


_INHERIT_PREPROCESSORS: _Inherit = _Inherit()


# PUBLIC API:


def pipeline(preprocessors: InputPreprocessor = ()) -> Pipeline:
    """Convenience function for `Pipeline` collection."""
    return Pipeline(preprocessors)


def col(columns: Columns, /, *, preprocessors: InputPreprocessor = ()) -> Col:
    """Convenience function for calling `Col` with a deduper."""
    return Col(columns, preprocessors=preprocessors)


# COL:


@final
class Col:
    """Unit collection for a single deduper in the Pipeline API.

    Operates deduplication on a column. Passed as a step in a deduplication
    pipeline. Allows for usage of dedupers as method calls instead of functions.

    Args:
        columns: the label(s) of a column or columns.

    Example:
        single `on` in a pipeline step:

            import liken as lk

            lk.pipeline().step(lk.col("address").exact())

        Dedupers are combined as "and" statements when passed as a members
        of a list in a step:

            lk.pipeline().step(
                [
                    lk.col("email").fuzzy(threshold=0.95),
                    lk.col("email").str_endswith("UK"),
                ]
            )

        Dedupers can be combined as "and" statement in a step for
        **different** columns:

            lk.pipeline().step(
                [
                    lk.col("email").fuzzy(threshold=0.95),
                    ~lk.col("address").isna(),
                ]
            )

        Pipeline preprocessors will be applied to all dedupers:

            pipeline = (
                lk.pipeline(preprocessors=[lk.preprocessors.lower()])
                .step(lk.col("email").fuzzy()) # Preprocessor applies here
                .step(lk.col("address").tfidf()) # And here
            )

        But, a global pipeline preprocessor will not override an explicit
        deduper's preprocessor. Similarly, a step's preprocessor will not
        override the deduper's:

            pipeline = (
                lk.pipeline(preprocessors=[lk.preprocessors.ascii_fold()])
                .step(
                    [
                        lk.col("email").fuzzy(),  # preprocessed by step's preprocessor, `alnum`.
                        ~lk.col(
                            "address",
                            preprocessors=[lk.preprocessors.lower()],
                        ).isna(), # uses it's own preprocessor, `lower`.
                    ],
                    preprocessors=[lk.preprocessors.alnum()], # defines the step's preprocessor
                )
                .step(lk.col("address").tfidf()) # defaults to the pipeline's preprocessor, `ascii_fold`.
            )
    """

    # for IDE autocompletion only!
    # must be manually maintained
    # add a new dummy method here upon adding a new deduper.

    def exact(self, *args, **kwargs) -> Col:
        """Method wrapper of `lk.exact` function.

        Usage is identical to the function but chained to an instance of `Col`.

        Info:
            See [`lk.exact`][liken.exact] reference for complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(lk.col("col").exact())

        """
        return self.__getattr__("exact")(*args, **kwargs)

    def fuzzy(self, *args, **kwargs) -> Col:
        """Method wrapper of `lk.fuzzy` function.

        Usage is identical to the function but chained to an instance of `Col`.

        Info:
            See [`lk.fuzzy`][liken.fuzzy] reference for complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(lk.col("col").fuzzy())

        """
        return self.__getattr__("fuzzy")(*args, **kwargs)

    def lsh(self, *args, **kwargs) -> Col:
        """Method wrapper of `lk.lsh` function.

        Usage is identical to the function but chained to an instance of `Col`.

        Info:
            See [`lk.lsh`][liken.lsh] reference for complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(lk.col("col").lsh())

        """
        return self.__getattr__("lsh")(*args, **kwargs)

    def tfidf(self, *args, **kwargs) -> Col:
        """Method wrapper of `lk.tfidf` function.

        Usage is identical to the function but chained to an instance of `Col`.

        Info:
            See [`lk.tfidf`][liken.tfidf] reference for complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(lk.col("col").tfidf())

        """
        return self.__getattr__("tfidf")(*args, **kwargs)

    def edit_distance(self, *args, **kwargs) -> Col:
        """Method wrapper of `lk.edit_distance` function.

        Usage is identical to the function but chained to an instance of `Col`.

        Info:
            See [`lk.edit_distance`][liken.edit_distance] reference for complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(lk.col("col").edit_distance())

        """
        return self.__getattr__("edit_distance")(*args, **kwargs)

    def cosine(self, *args, **kwargs) -> Col:
        """Method wrapper of `lk.cosine` function.

        Usage is identical to the function but chained to an instance of `Col`.

        Info:
            See [`lk.cosine`][liken.cosine] reference for complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(lk.col("col").cosine())

        """
        return self.__getattr__("cosine")(*args, **kwargs)

    def jaccard(self, *args, **kwargs) -> Col:
        """Method wrapper of `lk.jaccard` function.

        Usage is identical to the function but chained to an instance of `Col`.

        Info:
            See [`lk.jaccard`][liken.jaccard] reference for complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(lk.col("col").jaccard())

        """
        return self.__getattr__("jaccard")(*args, **kwargs)

    def token_overlap(self, *args, **kwargs) -> Col:
        """Method wrapper of `lk.token_overlap` function.

        Usage is identical to the function but chained to an instance of `Col`.

        Info:
            See [`lk.token_overlap`][liken.token_overlap] reference for complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(lk.col("col").token_overlap())

        """
        return self.__getattr__("token_overlap")(*args, **kwargs)

    def isin(self, *args, **kwargs) -> Col:
        """Method wrapper of `lk.isin` function.

        Usage is identical to the function but chained to an instance of `Col`.

        Info:
            See [`lk.isin`][liken.isin] reference for complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(lk.col("col").isin())

        """
        return self.__getattr__("isin")(*args, **kwargs)

    def isna(self, *args, **kwargs) -> Col:
        """Method wrapper of `lk.isna` function.

        Usage is identical to the function but chained to an instance of `Col`.

        Info:
            See [`lk.isna`][liken.isna] reference for complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(lk.col("col").isna())

        """
        return self.__getattr__("isna")(*args, **kwargs)

    def num_range(self, *args, **kwargs) -> Col:
        """Method wrapper of `lk.num_range` function.

        Usage is identical to the function but chained to an instance of `Col`.

        Info:
            See [`lk.num_range`][liken.num_range] reference for complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(lk.col("col").num_range())

        """
        return self.__getattr__("num_range")(*args, **kwargs)

    def within(self, *args, **kwargs) -> Col:
        """Method wrapper of `lk.within` function.

        Usage is identical to the function but chained to an instance of `Col`.

        Info:
            See [`lk.within`][liken.within] reference for complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(lk.col("col").within(tol=0.01))

        """
        return self.__getattr__("within")(*args, **kwargs)

    def date_range(self, *args, **kwargs) -> Col:
        """Method wrapper of `lk.date_range` function.

        Usage is identical to the function but chained to an instance of `Col`.

        Info:
            See [`lk.date_range`][liken.date_range] reference for complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(lk.col("col").date_range(format="%Y-%m-%d", window=1))

        """
        return self.__getattr__("date_range")(*args, **kwargs)

    def str_contains(self, *args, **kwargs) -> Col:
        """Method wrapper of `lk.str_contains` function.

        Usage is identical to the function but chained to an instance of `Col`.

        Info:
            See [`lk.str_contains`][liken.str_contains] reference for complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(lk.col("col").str_contains())

        """
        return self.__getattr__("str_contains")(*args, **kwargs)

    def str_endswith(self, *args, **kwargs) -> Col:
        """Method wrapper of `lk.str_endswith` function.

        Usage is identical to the function but chained to an instance of `Col`.

        Info:
            See [`lk.str_endswith`][liken.str_endswith] reference for complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(lk.col("col").str_endswith())

        """
        return self.__getattr__("str_endswith")(*args, **kwargs)

    def str_len(self, *args, **kwargs) -> Col:
        """Method wrapper of `lk.str_len` function.

        Usage is identical to the function but chained to an instance of `Col`.

        Info:
            See [`lk.str_len`][liken.str_len] reference for complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(lk.col("col").str_len())

        """
        return self.__getattr__("str_len")(*args, **kwargs)

    def str_startswith(self, *args, **kwargs) -> Col:
        """Method wrapper of `lk.str_startswith` function.

        Usage is identical to the function but chained to an instance of `Col`.

        Info:
            See [`lk.str_startswith`][liken.str_startswith] reference for complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(lk.col("col").str_startswith())

        """
        return self.__getattr__("str_startswith")(*args, **kwargs)

    def sorted_tokens(self, *args, **kwargs) -> Col:
        """Method wrapper of `lk.sorted_tokens` function.

        Usage is identical to the function but chained to an instance of `Col`.

        Info:
            See [`lk.sorted_tokens`][liken.sorted_tokens] reference for complete documentation.

        Example:
            Define as part of a pipeline:

                pipeline = lk.pipeline().step(lk.col("col").sorted_tokens())

        """
        return self.__getattr__("sorted_tokens")(*args, **kwargs)

    def __init__(self, columns: Columns, preprocessors: InputPreprocessor = ()):
        self._columns: Columns = columns
        self._unit: PipelineUnit
        self._used_in_step: bool = False
        self._preprocessors: list[Preprocessor] = resolve_preprocessors(preprocessors)

    def __getattr__(self, attr: str) -> Callable[..., Col]:
        """Make deduper functions available as method calls to Col.

        Functions are retrieved from registry. Includes any prior custom
        dedupers that have been registered.
        """

        # No intercept: Python internals
        if attr.startswith("__"):
            raise AttributeError(attr)

        # An unset `_unit` is a chaining error, not a registry lookup.
        if attr == "_unit":
            raise InvalidDeduperError(INVALID_NO_DEDUPER_CHAINED_MSG)

        func = dedupers_registry.get(f"{attr}")

        def wrapper(*args: Any, **kwargs: Any) -> Col:
            deduper = func(*args, **kwargs)
            deduper.validate(self._columns)
            if "_unit" in self.__dict__:
                # chaining a second deduper keeps last-wins
                warn(WARN_COL_RECHAINED_MSG)
            self._unit = PipelineUnit(self._columns, deduper, self._preprocessors)
            return self

        return wrapper

    def __invert__(self) -> Col:
        """Propagate inversion to the deduper. Allows for following syntax:

        ~lk.col("email").isna()

        Where the inversion gets propagated to act on isna().
        """

        columns, deduper, preprocessors = self._unit

        # `require_type` raises for a non-predicate deduper; the cast makes
        # the guarantee visible to mypy for the inversion below.
        deduper = cast("PredicateDeduper", require_type("deduper", deduper, PredicateDeduper))

        new_on = Col(columns)
        new_on._unit = PipelineUnit(columns, ~deduper, preprocessors)
        return new_on

    @property
    def unit(self) -> PipelineUnit:
        unit = self.__dict__.get("_unit")
        if unit is None:
            raise InvalidDeduperError(INVALID_NO_DEDUPER_CHAINED_MSG)
        return unit

    def __str__(self) -> str:
        """string representation of a single chained Col"""
        on: str = "lk.col"

        rep = ""
        columns, deduper, preprocessors = self._unit
        deduper_str: str = cast("str", str(deduper))
        if deduper_str.startswith("~"):
            deduper_str = deduper_str[1:]
            on = "~" + on
        col_str = f"{on}('{columns}')"
        if preprocessors:
            col_str = f"{on}('{columns}', preprocessors={[str(p) for p in preprocessors]})"
        rep += f"{col_str}.{deduper_str}"
        return rep


# PIPELINE:


class Pipeline:
    """Builds deduplication pipeline.

    Defines deduplication steps as chainable calls. Accepts preprocessors.
    Preprocessors are propagate to each step and on each column, unless
    explicit overridden in those stages.

    Args:
        preprocessors: a preprocessor, or list of preprocessors to apply to the
            pipeline

    Raises:
        TypeError: if passed preprocessor not a member of `liken.preprocessors`
    """

    def __init__(self, preprocessors: InputPreprocessor = ()):
        self._preprocessors: list[Preprocessor] = resolve_preprocessors(preprocessors)
        self._cols: list[list[Col]] = []
        self._steps: PipelineCollection = []

    def step(
        self,
        cols: Col | list[Col],
        /,
        *,
        preprocessors: InputPreprocessor | None | _Inherit = _INHERIT_PREPROCESSORS,
    ) -> Self:
        """Define a deduplication pipeline step

        Accepts a single deduper or a list of dedupers. A list of dedupers that
        has more than one member are composed as "and rules" meaning that the
        deduplication of both (or more) dedupers are considered for
        canonicalisation.

        Args:
            cols: `Col` deduper, or list of the same.
            preprocessors: a preprocessor, or list of preprocessors to apply the
                whole step (i.e. to all dedupers if more than one deduper).
                Omitted, the step inherits the pipeline's preprocessors. An
                empty list or `None` means the step uses no preprocessors.

        Returns:
            Self

        Example:
            A single deduper is added:

                pipeline = lk.pipeline().step(lk.col("email").exact())

            Multiple steps can be chained:

                pipeline = (
                    lk.pipeline()
                    .step(lk.col("email").fuzzy())
                    .step(lk.col("address").tfidf())
                )

            More than on deduper can be added to form composable rules within
            a step:

                pipeline = (
                    lk.pipeline()
                    .step(
                        [
                            lk.col("email").fuzzy(),
                            ~lk.col("address").isna(),
                        ]
                    )
                    .step(lk.col("address").tfidf())
                )

            Pipeline preprocessors will be applied to all steps:

                pipeline = (
                    lk.pipeline(preprocessors=[lk.preprocessors.lower()])
                    .step(lk.col("email").fuzzy()) # Preprocessor applies here
                    .step(lk.col("address").tfidf()) # And here
                )

            But, a global pipeline preprocessor will not override an explicit
            step's preprocessor

                pipeline = (
                    lk.pipeline(preprocessors=[lk.preprocessors.lower()])
                    .step(
                        lk.col("email").fuzzy(), preprocessors=[lk.preprocessors.alnum()]
                    )  # only `alnum` preprocessed
                    .step(lk.col("address").tfidf())  # this one still preprocessed with `lower`
                )
        """
        if isinstance(preprocessors, _Inherit):
            preprocessors_list: list[Preprocessor] = self._preprocessors
        else:
            preprocessors_list = resolve_preprocessors(preprocessors)

        if isinstance(cols, list):
            if not cols:
                raise InvalidArgumentValueError(INVALID_EMPTY_PIPELINE_STEP_MSG)
            for member in cols:
                require_type("cols", member, Col)
            cols_list = cols
        else:
            cols_list: list[Col] = [require_type("cols", cols, Col)]

        # reusing one Col across steps warns.
        for col_unit in cols_list:
            if col_unit._used_in_step:
                warn(WARN_COL_REUSED_MSG)
            col_unit._used_in_step = True

        self._cols.append(cols_list)

        step: PipelineStep = [col.unit for col in cols_list]

        # propagate preprocessor
        step = [s._replace(preprocessors=preprocessors_list) if not s.preprocessors else s for s in step]

        # predicates sorted to first for "rule predication"
        step: PipelineStep = sorted(step, key=lambda x: not isinstance(x[1], PredicateDeduper))

        self._steps.append(step)

        return self

    def __str__(self) -> str:
        pros = ""
        if preprocessors := self._preprocessors:
            pros = "preprocessors=" + f"{[str(p) for p in preprocessors]}"

        inner = ""
        for step in self._cols:
            inner += "\n\t\t" + ".step(["
            for on in step:
                inner += "\n\t\t\t" + str(on) + ","
            inner += "\n\t\t" + "])"
        return f"(\n\tlk.pipeline({pros}){inner}\n)"

    @staticmethod
    def _has_any_predicate(step: list[tuple[Columns, BaseDeduper]]) -> bool:
        """Retrieve whether or not the pipeline step has at least one predicate
        deduper.

        Private! not public API.
        """
        return any(isinstance(x[1], PredicateDeduper) for x in step)

    @property
    def steps(self):
        return self._steps


# HELPERS:


def resolve_preprocessors(
    preprocessors: InputPreprocessor | None,
) -> list[Preprocessor]:
    if preprocessors is None:
        return []
    if isinstance(preprocessors, (list, tuple)):
        return [require_type("preprocessor", p, Preprocessor) for p in preprocessors]
    # i.e. a single preprocessor
    return [require_type("preprocessor", preprocessors, Preprocessor)]
