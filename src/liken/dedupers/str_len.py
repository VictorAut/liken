"""str len predicate deduper"""

from typing import ClassVar
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


@final
class StrLen(
    SingleColumnMixin,
    PredicateDeduper,
):
    """
    Deduplicates all instances of strings whose length satisfies the inclusive
    interval [min_len, max_len], so `min_len == max_len` selects the exact
    length. The empty string never satisfies the predicate, whatever the
    bounds; it matches only under `~str_len`. The upper bound can be left
    unbounded.
    """

    _NAME: ClassVar[str] = "str_len"

    def __init__(self, min_len: int = 0, max_len: int | None = None):
        super().__init__(min_len=min_len, max_len=max_len)
        self._min_len = min_len
        self._max_len = max_len

    @override
    def _matches(self, value: object) -> bool:
        # A missing value and the empty string never satisfy the predicate.
        if is_missing(value) or value == "":
            return False

        length = len(cast("str", value))

        if length < self._min_len:
            return False

        return not (self._max_len is not None and length > self._max_len)

    @override
    def _vectorized_matches(self, array: pa.Array) -> pa.Array:
        lengths = pc.utf8_length(array)

        # Base condition: min_len <= length
        mask = pc.greater_equal(lengths, self._min_len)

        if self._max_len is not None:
            upper = pc.less_equal(lengths, self._max_len)
            mask = pc.and_(mask, upper)

        # The empty string never satisfies the predicate, whatever the bounds.
        mask = pc.and_(mask, pc.greater(lengths, 0))

        # A missing value never satisfies the predicate, on either polarity.
        # Null comparisons stay null and `indices_nonzero` skips them, but the
        # mask's missing positions must stay null rather than collapse to
        # False: inverting a False would match a null on the negated path.
        # `pc.equal(array, array)` is True for a value and null for a null,
        # so composing it keeps every missing position null in the mask.
        missing = pc.invert(pc.equal(array, array))

        return pc.if_else(missing, pa.scalar(None), mask)

    def __str__(self):
        return self.str_representation(self._NAME)


@dedupers_registry.register("str_len")
def str_len(min_len: int = 0, max_len: int | None = None) -> BaseDeduper:
    """Discrete deduper on string length.

    Usage is on a single column of a dataframe. Available as the inversion, i.e.
    "not the defined length" using inversion operator: `~str_len()`.

    Deduplication will happen over the lengths inside the inclusive interval
    [min_len, max_len]. A value whose length equals either bound matches, so
    `min_len == max_len` selects the exact length. The upper end of the range
    can be left unbounded. The empty string never satisfies the predicate,
    whatever the bounds; it matches only under `~str_len()`. All matched rows
    collapse into one group. A missing value never satisfies the predicate,
    on either polarity.

    Args:
        min_len: the inclusive lower bound of the interval.
        max_len: the inclusive upper bound of the interval. `None` leaves the
            interval unbounded above.

    Returns:
        Instance of `BaseDeduper`.

    Example:
        Applied to a single column:

            import liken as lk

            pipeline = lk.pipeline().step(
                [
                    lk.col("email").exact(),
                    lk.col("email").str_len(min_len=10),
                ]
            )

            df = (
                lk.dedupe(df)
                .apply(pipeline)
                .drop_duplicates(keep="last")
            )

            >>> df # before
            +------+-----------+---------------------+
            | id   |  address  |         email       |
            +------+-----------+---------------------+
            |  1   |  london   |  fizzpop@yahoo.com  |
            |  2   |   tokyo   |  fizzpop@yahoo.com  |
            |  3   |   paris   |       a@msn.fr      |
            |  4   |   nice    |       a@msn.fr      |
            +------+-----------+---------------------+

            >>> df # after
            +------+-----------+---------------------+
            | id   |  address  |         email       |
            +------+-----------+---------------------+
            |  2   |   tokyo   |  fizzpop@yahoo.com  |
            |  3   |   paris   |       a@msn.fr      |
            |  4   |   nice    |       a@msn.fr      |
            +------+-----------+---------------------+
    """
    return StrLen(min_len=min_len, max_len=max_len)
