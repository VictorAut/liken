"""Shared missing-value helper defining the library's null contract"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence


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


def partition_missing(values: Sequence[object]) -> tuple[list[int], list[int]]:
    """Split values into the indices of missing and non-missing values.

    Missing values are classified by :func:`is_missing`. Order is preserved
    within each partition, so each index list is ascending.

    Args:
        values: The values to partition.

    Returns:
        A tuple of two lists: the indices of the missing values, then the
        indices of the non-missing values.

    Example:
        >>> partition_missing([None, "a", float("nan"), "b"])
        ([0, 2], [1, 3])
    """
    missing: list[int] = []
    present: list[int] = []
    for i, value in enumerate(values):
        if is_missing(value):
            missing.append(i)
        else:
            present.append(i)
    return missing, present
