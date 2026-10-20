"""within deduper"""

from collections.abc import Iterator
from typing import ClassVar
from typing import Final
from typing import final

import pyarrow as pa

from liken.core.deduper import BaseDeduper
from liken.core.deduper import SingleColumnMixin
from liken.core.missing import partition_missing
from liken.core.registries import dedupers_registry
from liken.exceptions import InvalidArgumentTypeError
from liken.exceptions import InvalidArgumentValueError
from liken.types import SimilarPairIndices
from liken.validations import require_real_number


# Floating-point subtraction at the tolerance boundary can round above
# `tol` (`1.02 - 1.01` is `0.010000000000000009`), so the bound is widened
# by the guard below, mirroring `edit_distance`'s normalized metrics. The
# guard applies only when `tol > 0`: at `tol=0` equal values subtract to
# exactly `0.0`, so the match is exact equality with no rounding hazard.
_BOUNDARY_TOLERANCE: Final[float] = 1e-9


@final
class Within(
    SingleColumnMixin,
    BaseDeduper,
):
    """
    Numeric tolerance deduper.

    Two numbers are matched when the absolute difference between them is at
    most `tol`. The bound is inclusive. Missing values match only each
    other, never a value.

    Matching is transitive through union-find: under `tol=0.01` the chain
    1.0, 1.01, 1.02 merges into one group even though 1.0 and 1.02 differ
    by more than `tol`. The tolerance is absolute, not relative: it does
    not scale with the magnitude of the values.
    """

    _NAME: ClassVar[str] = "within"

    def __init__(self, tol: float):
        tol = require_real_number("tol", tol, low=0, low_inclusive=True)
        if tol == float("inf"):
            raise InvalidArgumentValueError("`tol` must be a finite number, got inf")
        super().__init__(tol=tol)
        self._tol = tol

    def _gen_similarity_pairs(self, array: pa.Array) -> Iterator[SimilarPairIndices]:
        # An all-empty array arrives typed `null`; with no values there is
        # nothing to match, so the numeric check passes it through.
        numeric = (
            array.type == pa.null()
            or pa.types.is_integer(array.type)
            or pa.types.is_floating(array.type)
            or pa.types.is_decimal(array.type)
        )
        if not numeric:
            raise InvalidArgumentTypeError(f"`within` requires a numeric column, got Arrow type '{array.type}'")

        values: list = array.to_pylist()

        missing, present = partition_missing(values)

        for i in missing[1:]:
            yield missing[0], i

        if not present:
            return

        # Sort once, then slide a window over the sorted order: a pair
        # within `tol` in either direction is adjacent in sorted order
        # within `tol`. O(n log n) overall; no all-pairs scan.
        order = sorted(present, key=lambda i: values[i])

        guard = _BOUNDARY_TOLERANCE if self._tol > 0 else 0.0

        # `float("inf")` data values: two infinities subtract to NaN, and
        # a NaN comparison is always False, so the bound is written as
        # `not (diff <= bound)` — a NaN difference falls outside the
        # window and infinities never match. `start` is capped at `pos`:
        # in sorted order every value before an infinity is an infinity
        # too, so an infinity pairs with nothing, itself included.
        start = 0
        for pos in range(1, len(order)):
            left = order[pos]
            while start < pos and not abs(values[left] - values[order[start]]) <= self._tol + guard:
                start += 1
            for j in range(start, pos):
                i = order[j]
                if i < left:
                    yield i, left
                else:
                    yield left, i

    def __str__(self):
        return self.str_representation(self._NAME)


@dedupers_registry.register("within")
def within(tol: float) -> BaseDeduper:
    """Numeric tolerance deduplication.

    Usage is on single columns of a dataframe.

    Two numbers are matched when the absolute difference between them is at
    most `tol`. The bound is inclusive: a pair differing by exactly `tol`
    matches. `tol=0` reduces to numeric equality. The tolerance is
    absolute: it does not scale with the magnitude of the values.

    Matching is transitive: values that connect through a chain of pairs
    merge into one group. Under `tol=0.01`, the chain 1.0, 1.01, 1.02
    merges even though 1.0 and 1.02 differ by more than `tol`. Wide
    tolerances relative to the spread of the data can therefore collapse
    many rows into one group.

    Missing values, `None` or NaN, match only each other, never a value.

    Args:
        tol: The maximum absolute difference at which two values are
            considered valid for deduplication. Must be a finite,
            non-negative real number; there is no default.

    Returns:
        Instance of `BaseDeduper`.

    Raises:
        InvalidArgumentTypeError: if `tol` is not a real number, or if the
            column is not numeric at execution.
        InvalidArgumentValueError: if `tol` is negative, NaN or infinite.

    Example:
        Applied to a single column:

            import liken as lk

            df = (
                lk.dedupe(df)
                .apply({"price": within(tol=0.05)})
                .drop_duplicates()
            )

        E.g.

            >>> df # Before
            +------+-------+----------------------+
            | id   | price |         email        |
            +------+-------+----------------------+
            |  1   | 10.02 |  fizzpop@gmail.com   |
            |  2   |  null |  foobar@gmail.com    |
            |  3   | 10.03 |  foobar@gmail.co.uk  |
            +------+-----------+------------------+

            >>> df # After
            +------+-------+----------------------+
            | id   | price |         email        |
            +------+-------+----------------------+
            |  1   | 10.02 |  fizzpop@gmail.com   |
            |  3   | 10.03 |  foobar@gmail.co.uk  |
            +------+-----------+------------------+
    """
    return Within(tol=tol)
