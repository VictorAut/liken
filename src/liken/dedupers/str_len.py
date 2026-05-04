"""str len predicate deduper"""

from typing import ClassVar
from typing import final

import pyarrow as pa
import pyarrow.compute as pc
from typing_extensions import override

from liken.core.deduper import BaseDeduper
from liken.core.deduper import PredicateDeduper
from liken.core.deduper import SingleColumnMixin
from liken.core.registries import dedupers_registry


@final
class StrLen(
    SingleColumnMixin,
    PredicateDeduper,
):
    """
    Deduplicates all instances of strings that satisfy the bounds in
    (min_len, max_len) where the upper bound can actually be left unbounded.
    """

    _NAME: ClassVar[str] = "str_len"

    def __init__(self, min_len: int = 0, max_len: int | None = None):
        super().__init__(min_len=min_len, max_len=max_len)
        self._min_len = min_len
        self._max_len = max_len

    @override
    def _vectorized_matches(self, array: pa.Array) -> pa.Array:
        lengths = pc.utf8_length(array)

        # Base condition: length > min_len
        mask = pc.greater(lengths, self._min_len)

        if self._max_len is not None:
            upper = pc.less_equal(lengths, self._max_len)
            mask = pc.and_(mask, upper)

        # Exclude nulls / empty strings
        not_null = pc.invert(pc.is_null(array))
        not_empty = pc.greater(lengths, 0)

        mask = pc.and_(mask, not_null)
        mask = pc.and_(mask, not_empty)

        return mask

    def __str__(self):
        return self.str_representation(self._NAME)


@dedupers_registry.register("str_len")
def str_len(min_len: int = 0, max_len: int | None = None) -> BaseDeduper:
    """Discrete deduper on string length.

    Usage is on a single column of a dataframe. Available as the inversion, i.e.
    "not the defined length" using inversion operator: `~str_len()`.

    Deduplication will happen over the bounded lengths defined by `min_len` and
    `max_len`. The upper end of the range can be left unbounded. For
    deduplication over an exact length use `max_len = min_len + 1`.

    Args:
        min_len: the lower bound of lengths considered
        max_len: the upper bound of lengths considered. Can be left unbounded.

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
