"""Shared missing-value helper defining the library's null contract"""

from __future__ import annotations


def is_missing(value: object) -> bool:
    """Report whether a value is missing.

    A value is missing when it is ``None`` or not equal to itself. The latter
    covers every IEEE NaN (float NaN, ``Decimal("NaN")``) by IEEE semantics,
    so ``None`` and NaN are one missing class.

    This is the one definition of missing for the library. Dedupers,
    predicates and wrappers must use it rather than implementing their own
    missing check.

    Args:
        value: The value to test.

    Returns:
        True when the value is missing, False otherwise.

    Example:
        >>> is_missing(None)
        True
        >>> is_missing(float("nan"))
        True
        >>> is_missing("na")
        False
    """
    return value is None or value != value
