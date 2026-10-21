"""date range predicate deduper"""

from collections.abc import Iterator
from typing import ClassVar
from typing import final

import pyarrow as pa
import pyarrow.compute as pc
from typing_extensions import override

from liken.core.deduper import BaseDeduper
from liken.core.deduper import PredicateDeduper
from liken.core.deduper import SingleColumnMixin
from liken.core.missing import is_missing
from liken.core.missing import partition_missing
from liken.core.registries import dedupers_registry
from liken.exceptions import InvalidArgumentTypeError
from liken.exceptions import InvalidArgumentValueError
from liken.types import SimilarPairIndices
from liken.validations import require_int
from liken.validations import require_non_empty_str


_SECONDS_PER_DAY: int = 86_400


@final
class DateRange(
    SingleColumnMixin,
    PredicateDeduper,
):
    """
    Deduplicates records whose parsed dates are at most `window` days apart.

    Every value parses under the explicit `format`; the deduper never
    guesses. Matching is on day granularity: the time-of-day part of a
    format that carries one is ignored, and timezones are out of scope.
    Missing values match only each other, never a value.

    Inversion follows the per-row predicate contract: `_matches` is "the
    value parses under `format`", so `~date_range` groups values that fail
    to parse, and a missing value matches nothing on either polarity.
    """

    _NAME: ClassVar[str] = "date_range"

    def __init__(self, format: str, window: int = 0):
        format = require_non_empty_str("format", format)
        window = require_int("window", window, low=0)

        super().__init__(format=format, window=window)
        self._format = format
        self._window = window

        # Column label, stashed by `validate` for run-time error messages.
        # `build_union_find` always calls `validate` before pair generation.
        self._validated_column: str | None = None

    @override
    def validate(self, columns) -> None:
        super().validate(columns)
        self._validated_column = columns

    @override
    def _matches(self, value: object) -> bool:
        # "The value parses under `format`": the row participates in date
        # matching. A missing value never participates. Only the negated
        # path consumes this — the positive path is pairwise (a date
        # matches relative to its neighbours) and overrides
        # `_gen_similarity_pairs`.
        if is_missing(value):
            return False
        try:
            pc.strptime(pa.array([value]), format=self._format, unit="s")
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            return False
        return True

    def _column(self) -> str:
        """Column label for error messages; a defensive fallback for direct
        calls that skipped `validate`."""
        return self._validated_column if self._validated_column is not None else "?"

    def _day_ordinals(self, array: pa.Array) -> pa.Array:
        """Parse the string array to integer day ordinals (days since the
        epoch, signed). Nulls pass through as nulls.

        Raises:
            InvalidArgumentValueError: if a value fails to parse, naming
                the column and the format.
        """
        try:
            parsed = pc.strptime(array, format=self._format, unit="s")
        except pa.ArrowInvalid as err:
            raise InvalidArgumentValueError(
                f"column '{self._column()}': values do not parse under format '{self._format}': {err}"
            ) from err

        # Floor to whole days before the integer cast: a raw cast to int64
        # truncates toward the epoch and shifts pre-epoch timestamps by a
        # day. Time-of-day is then divided away; day granularity.
        return pc.divide(pc.floor_temporal(parsed, unit="day").cast(pa.int64()), pa.scalar(_SECONDS_PER_DAY))

    @override
    def _gen_similarity_pairs(self, array: pa.Array) -> Iterator[SimilarPairIndices]:
        # The all-empty array arrives typed `null`; with no values there is
        # nothing to parse, so the string check passes it through.
        if array.type != pa.null() and not (pa.types.is_string(array.type) or pa.types.is_large_string(array.type)):
            raise InvalidArgumentTypeError(f"`date_range` requires a string column, got Arrow type '{array.type}'")

        values: list = array.to_pylist()

        missing, present = partition_missing(values)

        for i in missing[1:]:
            yield missing[0], i

        if not present:
            return

        ordinals = self._day_ordinals(array.take(pa.array(present)))
        days: list = ordinals.to_pylist()

        # Bucket day ordinals into cells of width window + 1. Two dates
        # within `window` days always fall in the same or adjacent cells,
        # and cells two or more apart are always more than `window` days
        # apart, so only the successor cell needs the join. At window=0 the
        # cells are single days and the neighbour join produces nothing:
        # window=0 degenerates to same-date bucketing.
        width = self._window + 1

        cells: dict[int, list[int]] = {}
        for pos, day in enumerate(days):
            cells.setdefault(day // width, []).append(pos)

        for cell, members in cells.items():
            # Same cell: any two ordinals inside differ by at most
            # `window` days, so the pair always matches. Star-shaped from
            # the cell's first index (predicate convention). Indices map
            # back through `present` to the original positions.
            first = members[0]
            for pos in members[1:]:
                a, b = (first, pos) if first < pos else (pos, first)
                yield present[a], present[b]

            # Neighbour cell: the day-distance test decides, because a
            # cell holds dates up to `window` days apart and only part of
            # the neighbour is within reach.
            neighbour = cells.get(cell + 1)
            if neighbour is not None:
                for pos in members:
                    for other in neighbour:
                        if abs(days[pos] - days[other]) <= self._window:
                            a, b = (pos, other) if pos < other else (other, pos)
                            yield present[a], present[b]

    def __str__(self):
        return self.str_representation(self._NAME)


@dedupers_registry.register("date_range")
def date_range(format: str, window: int = 0) -> BaseDeduper:
    """Date-bucket predicate deduplication.

    Usage is on a single string column of a dataframe. Available as the
    inversion using the inversion operator: `~date_range(...)`.

    Each value parses under the explicit `format` — a pyarrow strptime
    pattern such as `"%Y-%m-%d"`. The deduper never guesses the format:
    a value that does not parse raises an error naming the column and the
    format, rather than silently dropping the value. A value that parses
    under a different format is an error, not a match; mixed formats must
    be normalised before deduplication.

    Two values match when their dates differ by at most `window` days.
    The bound is inclusive: exactly `window` days apart matches.
    `window=0` reduces to same-date matching. Matching is on day
    granularity; a format carrying a time component contributes only its
    date part. Timezones are out of scope.

    Matching is transitive: dates that connect through a chain of pairs
    merge into one group.

    Missing values, `None`, match only each other, never a value. Under
    `~date_range` the groups invert: values that fail to parse group
    together, and a missing value still matches nothing.

    Args:
        format: The pyarrow strptime pattern every value must parse
            under. Required; the deduper never guesses.
        window: The maximum day distance at which two dates match.
            Must be a non-negative integer; defaults to 0 (same-date).

    Returns:
        Instance of `BaseDeduper`.

    Raises:
        InvalidArgumentTypeError: if `format` is not a string, if
            `window` is not an integer, or if the column is not a string
            column at execution.
        InvalidArgumentValueError: if `format` is empty, if `window` is
            negative, or if a value fails to parse under `format` at
            execution.

    Example:
        Applied to a single column:

            import liken as lk

            pipeline = lk.pipeline().step(
                [
                    lk.col("email").exact(),
                    lk.col("dob").date_range(format="%Y-%m-%d", window=1),
                ]
            )

            df = (
                lk.dedupe(df)
                .apply(pipeline)
                .drop_duplicates(keep="last")
            )

            >>> df # before
            +------+------------+---------------------+
            | id   |    dob     |        email        |
            +------+------------+---------------------+
            |  1   | 2024-01-05 |  fizzpop@yahoo.com  |
            |  2   | 2024-01-06 |  fizzpop@yahoo.com  |
            |  3   | 2024-02-28 |   foobar@gmail.com  |
            |  4   |     null   |   foobar@gmail.com  |
            |  5   |     null   |   foobar@gmail.com  |
            +------+------------+---------------------+

            >>> df # after
            +------+------------+---------------------+
            | id   |    dob     |        email        |
            +------+------------+---------------------+
            |  2   | 2024-01-06 |  fizzpop@yahoo.com  | # rows 1, 2 within 1 day
            |  3   | 2024-02-28 |   foobar@gmail.com  |
            |  5   |     null   |   foobar@gmail.com  | # nulls group together
            +------+------------+---------------------+
    """
    return DateRange(format=format, window=window)
