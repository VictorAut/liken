"""This module contains argument validation for classes.

Most validations are for public arguments of the 'Dedupe' class.

However, some validations exist for other private classes

argument type, shape and value checks run eagerly at
construction — including deduper knobs and preprocessor parameters, and the
column-arity mixins via `Col` chaining. Frame-dependent checks (column
existence, frame shape) run at the frame boundary. Checks that depend on the
data values themselves run at execution only.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING
from typing import Literal
from typing import TypeVar
from typing import cast

from liken.exceptions import InvalidArgumentTypeError
from liken.exceptions import InvalidArgumentValueError


if TYPE_CHECKING:
    from liken.core.deduper import BaseDeduper
    from liken.types import Columns


T = TypeVar("T")


def _is_real_number(value: object) -> bool:
    """True for int/float but not bool (which subclasses int)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _render_type(expected: type | tuple[type, ...]) -> str:
    """Render expected type(s) as a quoted name list."""
    if isinstance(expected, tuple):
        return " or ".join(f"'{t.__name__}'" for t in expected)
    return f"'{expected.__name__}'"


def _render_bound(low: float | None, high: float | None, low_inclusive: bool, high_inclusive: bool) -> str:
    """Render bounds in math interval notation; infinite ends stay open."""
    left = "[" if (low is not None and low_inclusive) else "("
    right = "]" if (high is not None and high_inclusive) else ")"
    low_s = "-inf" if low is None else repr(low)
    high_s = "inf" if high is None else repr(high)
    return f"{left}{low_s}, {high_s}{right}"


def require_type(name: str, value: T, expected: type[T] | tuple[type[T], ...]) -> T:
    """Require `value` to be an instance of `expected`.

    Args:
        name: argument name, used in the error message.
        value: the argument to validate.
        expected: the required type, or a tuple of allowed types.

    Returns:
        The validated value, unchanged.

    Raises:
        InvalidArgumentTypeError: if `value` is not an instance of
            `expected`.
    """
    if not isinstance(value, expected):
        got = type(value).__name__
        raise InvalidArgumentTypeError(f"`{name}` must be of type {_render_type(expected)}, got '{got}'")
    return value


def require_real_number(
    name: str,
    value: object,
    *,
    low: float | None = None,
    high: float | None = None,
    low_inclusive: bool = True,
    high_inclusive: bool = False,
    allow_none: bool = False,
) -> float:
    """Require `value` to be a real number, optionally within bounds.

    Accepts `int` and `float`, rejects `bool` (an `int` subclass) and `NaN`.
    Infinite values pass unless rejected by the bounds.

    Args:
        name: argument name, used in the error message.
        value: the argument to validate.
        low: optional lower bound.
        high: optional upper bound.
        low_inclusive: whether `low` itself is accepted.
        high_inclusive: whether `high` itself is accepted.
        allow_none: return `None` unchanged instead of validating.

    Returns:
        The validated value.

    Raises:
        InvalidArgumentTypeError: if `value` is not a real number.
        InvalidArgumentValueError: if `value` is `NaN` or outside the bounds.
    """
    if allow_none and value is None:
        return cast("float", value)
    if not _is_real_number(value):
        raise InvalidArgumentTypeError(f"`{name}` must be a real number, got '{type(value).__name__}'")
    number = cast("float", value)
    if isinstance(number, float) and math.isnan(number):
        raise InvalidArgumentValueError(f"`{name}` must be a real number, got NaN")
    if low is not None and (number < low or (not low_inclusive and number == low)):
        raise InvalidArgumentValueError(
            f"`{name}` must be a real number in the range {_render_bound(low, high, low_inclusive, high_inclusive)},"
            f" got {number!r}"
        )
    if high is not None and (number > high or (not high_inclusive and number == high)):
        raise InvalidArgumentValueError(
            f"`{name}` must be a real number in the range {_render_bound(low, high, low_inclusive, high_inclusive)},"
            f" got {number!r}"
        )
    return number


def require_int(
    name: str,
    value: object,
    *,
    low: int | None = None,
    high: int | None = None,
    allow_none: bool = False,
) -> int:
    """Require `value` to be an `int`, optionally within bounds.

    Rejects `bool` (an `int` subclass), floats and numpy integer scalars.

    Args:
        name: argument name, used in the error message.
        value: the argument to validate.
        low: optional lower bound (inclusive).
        high: optional upper bound (inclusive).
        allow_none: return `None` unchanged instead of validating.

    Returns:
        The validated value.

    Raises:
        InvalidArgumentTypeError: if `value` is not an `int`.
        InvalidArgumentValueError: if `value` is outside the bounds.
    """
    if allow_none and value is None:
        return cast("int", value)
    if not isinstance(value, int) or isinstance(value, bool):
        raise InvalidArgumentTypeError(f"`{name}` must be an integer, got '{type(value).__name__}'")
    number = cast("int", value)
    if low is not None and number < low:
        raise InvalidArgumentValueError(
            f"`{name}` must be an integer in the range {_render_bound(low, high, True, True)}, got {number!r}"
        )
    if high is not None and number > high:
        raise InvalidArgumentValueError(
            f"`{name}` must be an integer in the range {_render_bound(low, high, True, True)}, got {number!r}"
        )
    return number


def require_non_empty_str(name: str, value: object, *, allow_none: bool = False) -> str:
    """Require `value` to be a non-empty string.

    Args:
        name: argument name, used in the error message.
        value: the argument to validate.
        allow_none: return `None` unchanged instead of validating.

    Returns:
        The validated value.

    Raises:
        InvalidArgumentTypeError: if `value` is not a `str`.
        InvalidArgumentValueError: if `value` is the empty string.
    """
    if allow_none and value is None:
        return cast("str", value)
    if not isinstance(value, str):
        raise InvalidArgumentTypeError(f"`{name}` must be of type 'str', got '{type(value).__name__}'")
    if not value:
        raise InvalidArgumentValueError(f"`{name}` must be a non-empty string, got {value!r}")
    return value


def require_choice(name: str, value: str, choices: tuple[str, ...]) -> str:
    """Require `value` to be one of `choices`.

    Args:
        name: argument name, used in the error message.
        value: the argument to validate.
        choices: the accepted values.

    Returns:
        The validated value, unchanged.

    Raises:
        InvalidArgumentValueError: if `value` is not in `choices`.
    """
    if value not in choices:
        raise InvalidArgumentValueError(f"`{name}` must be one of {choices!r}, got {value!r}")
    return value


def validate_keep_arg(keep: Literal["first", "last"]) -> Literal["first", "last"]:
    """Validates Keep arg in the 'Dedupe' class

    Raises:
        InvalidArgumentTypeError: if `keep` is not a string
        InvalidArgumentValueError: if `keep` is a string outside ("first", "last")
    """
    require_type("keep", keep, str)
    # `require_choice` returns `str`; the membership check above pins the
    # narrower `Literal` return type of this function.
    return cast("Literal['first', 'last']", require_choice("keep", keep, ("first", "last")))


def validate_strategy_arg(strategy: Literal["auto", "gram", "index"]) -> Literal["auto", "gram", "index"]:
    """Validates the set dedupers' `strategy` argument.

    Raises:
        InvalidArgumentTypeError: if `strategy` is not a string
        InvalidArgumentValueError: if `strategy` is a string outside
            ("auto", "gram", "index")
    """
    require_type("strategy", strategy, str)
    # `require_choice` returns `str`; the membership check above pins the
    # narrower `Literal` return type of this function.
    return cast(
        "Literal['auto', 'gram', 'index']",
        require_choice("strategy", strategy, ("auto", "gram", "index")),
    )


def validate_columns_arg(columns: Columns | None) -> Columns | None:
    """Validates the type and structure of the public api 'columns' arg.

    `columns` must be a column label or a tuple of column labels, with no
    repeated labels.

    Raises:
        InvalidArgumentTypeError: if `columns` is not a `str` or a tuple of
            `str`.
        InvalidArgumentValueError: if `columns` labels are repeated.
    """
    if columns is None:
        return None
    if isinstance(columns, list | set | frozenset | dict):
        raise InvalidArgumentTypeError(
            f"`columns` must be of type 'str' or 'tuple', got '{type(columns).__name__}'; use a tuple of column labels"
        )
    require_type("columns", columns, (str, tuple))
    if isinstance(columns, tuple):
        for label in columns:
            if not isinstance(label, str):
                raise InvalidArgumentTypeError(f"`columns` members must be of type 'str', got '{type(label).__name__}'")
        for label, count in Counter(columns).items():
            if count > 1:
                raise InvalidArgumentValueError(f"columns labels cannot be repeated. Repeated labels: '{label}'")
    return columns


def validate_thresholds_arg(thresholds: list[float]) -> list[float]:
    """Validates the `thresholds` arg of the 'Dedupe.explore' method.

    Must be a non-empty list of numbers, each in the open range (0, 1).

    Raises:
        InvalidArgumentTypeError: if `thresholds` is not a list, or a member
            is not a real number.
        InvalidArgumentValueError: if `thresholds` is empty, or a member is
            `NaN` or outside the range (0, 1).
    """
    require_type("thresholds", thresholds, list)
    if not thresholds:
        raise InvalidArgumentValueError("`thresholds` cannot be empty")
    for i, threshold in enumerate(thresholds):
        require_real_number(f"thresholds[{i}]", threshold, low=0, high=1, low_inclusive=False, high_inclusive=False)
    return thresholds


def validate_explore_columns_arg(
    columns: list[str] | dict[str, BaseDeduper],
) -> list[str] | dict[str, BaseDeduper]:
    """Validates the `columns` arg of the 'Dedupe.explore' method.

    Accepts either a non-empty list of column labels, or a non-empty dict
    mapping column labels to similarity (threshold) dedupers. Existence of the
    columns in the dataframe is checked separately, at execution time.

    Raises:
        InvalidArgumentValueError: if `columns` is neither a non-empty list
            of column labels nor a non-empty dict of single-column similarity
            dedupers.
    """
    # Function-level import: `core.deduper` imports the primitives from this
    # module, so a module-level import would be circular.
    from liken.core.deduper import CompoundColumnMixin
    from liken.core.deduper import ThresholdDeduper

    if isinstance(columns, list):
        if not columns or not all(isinstance(c, str) for c in columns):
            raise InvalidArgumentValueError(
                "columns for explore must be a non-empty list of column labels or a dict mapping columns to"
                f" dedupers, got {columns!r}"
            )
        return columns

    if isinstance(columns, dict):
        if not columns:
            raise InvalidArgumentValueError(
                "columns for explore must be a non-empty list of column labels or a dict mapping columns to"
                f" dedupers, got {columns!r}"
            )
        for label, deduper in columns.items():
            if not isinstance(label, str):
                raise InvalidArgumentValueError(
                    "columns for explore must be a non-empty list of column labels or a dict mapping columns to"
                    f" dedupers, got {columns!r}"
                )
            if not isinstance(deduper, ThresholdDeduper) or isinstance(deduper, CompoundColumnMixin):
                raise InvalidArgumentValueError(
                    f"explore dedupers must be single-column similarity (threshold) dedupers, got"
                    f" '{type(deduper).__name__}'"
                )
        return columns

    raise InvalidArgumentValueError(
        "columns for explore must be a non-empty list of column labels or a dict mapping columns to dedupers, got"
        f" {columns!r}"
    )


def validate_explore_column_exists(column: str, df_columns: list[str]) -> str:
    """Validates that an explored column exists in the dataframe.

    Raises:
        InvalidArgumentValueError: if the column is not in the dataframe.
    """
    if column not in df_columns:
        raise InvalidArgumentValueError(f"column '{column}' not found in the dataframe")
    return column
