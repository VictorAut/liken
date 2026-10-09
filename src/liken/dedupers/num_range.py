"""num range predicate deduper"""

from typing import ClassVar
from typing import TypeAlias
from typing import cast
from typing import final

import pyarrow as pa
import pyarrow.compute as pc
from typing_extensions import override

from liken.core.deduper import BaseDeduper
from liken.core.deduper import PredicateDeduper
from liken.core.deduper import SingleColumnMixin
from liken.core.missing import is_missing
from liken.core.registries import dedupers_registry


Number: TypeAlias = float | int


@final
class NumRange(
    SingleColumnMixin,
    PredicateDeduper,
):
    """
    Deduplicates all instances of numeric values that satisfy the inclusive
    interval [min, max]. Either bound may be left `None` (unbounded), but
    not both.
    """

    _NAME: ClassVar[str] = "num_range"

    def __init__(self, min: Number | None = None, max: Number | None = None):
        if min is None and max is None:
            raise ValueError("At least one of `min` or `max` must be given; a fully open range is refused")

        if min is not None and max is not None and min > max:
            raise ValueError("`min` must be less than or equal to `max`")

        super().__init__(min=min, max=max)
        self._min: Number | None = min
        self._max: Number | None = max

    @override
    def _matches(self, value: object) -> bool:
        # A missing value never satisfies the predicate.
        if is_missing(value):
            return False

        number = cast("Number", value)

        if self._min is not None and number < self._min:
            return False

        return not (self._max is not None and number > self._max)

    @override
    def _vectorized_matches(self, array: pa.Array) -> pa.Array:
        # At least one bound is given, enforced at construction.
        mask = pc.greater_equal(array, self._min) if self._min is not None else pc.less_equal(array, self._max)

        if self._min is not None and self._max is not None:
            mask = pc.and_(mask, pc.less_equal(array, self._max))

        # A missing value — null or NaN — never satisfies the predicate, on
        # either polarity. Null comparisons stay null and `indices_nonzero`
        # skips them, but a float NaN compares False and would be inverted
        # into a match on the negated path. `pc.equal(array, array)` is True
        # for a value, null for a null and False for a NaN, so composing it
        # makes every missing position null in the mask.
        missing = pc.invert(pc.equal(array, array))

        return pc.if_else(missing, pa.scalar(None), mask)

    def __str__(self):
        return self.str_representation(self._NAME)


@dedupers_registry.register("num_range")
def num_range(min: Number | None = None, max: Number | None = None) -> BaseDeduper:
    """Discrete deduper on a numeric interval.

    Usage is on a single column of a dataframe. Available as the inversion, i.e.
    "outside the interval" using inversion operator: `~num_range()`.

    Deduplication will happen over the values inside the inclusive interval
    [min, max]. Each bound applies only when given; one side may be left
    `None` (unbounded), but not both. All matched rows collapse into one
    group. A missing value — `None` or any IEEE NaN — never satisfies the
    predicate, on either polarity.

    Args:
        min: the inclusive lower bound of the interval. `None` leaves the
            interval unbounded below.
        max: the inclusive upper bound of the interval. `None` leaves the
            interval unbounded above.

    Returns:
        Instance of `BaseDeduper`.

    Raises:
        ValueError: if both bounds are `None` (a fully open range), or if
            `min > max` (an empty interval).

    Example:
        Applied to a single column:

            import liken as lk

            pipeline = lk.pipeline().step(
                [
                    lk.col("email").exact(),
                    lk.col("salary").num_range(min=30_000, max=40_000),
                ]
            )

            df = (
                lk.dedupe(df)
                .apply(pipeline)
                .drop_duplicates(keep="last")
            )

            >>> df # before
            +------+-----------------+---------+
            | id   |      email      |  salary |
            +------+-----------------+---------+
            |  1   |  fizz@yahoo.com |  32000  |
            |  2   |  fizz@yahoo.com |  51000  |
            |  3   |     a@msn.fr    |  30000  |
            |  4   |     a@msn.fr    |  40000  |
            |  5   |     a@msn.fr    |  99000  |
            +------+-----------------+---------+

            >>> df # after
            +------+-----------------+---------+
            | id   |      email      |  salary |
            +------+-----------------+---------+
            |  1   |  fizz@yahoo.com |  32000  |
            |  2   |  fizz@yahoo.com |  51000  |
            |  4   |     a@msn.fr    |  40000  | # rows 3 and 4 in band, merged
            |  5   |     a@msn.fr    |  99000  |
            +------+-----------------+---------+
    """
    return NumRange(min=min, max=max)
