"""Tests for the shared missing-value helper"""

from decimal import Decimal

import pytest

from liken.core.missing import is_missing
from liken.core.missing import partition_missing


@pytest.mark.parametrize("value", [None, float("nan"), Decimal("NaN")], ids=["none", "nan", "decimal-nan"])
def test_is_missing_true_for_missing_values(value):
    assert is_missing(value) is True


@pytest.mark.parametrize(
    "value",
    ["na", "", "None", "null", 0, False, 1.0, Decimal("1.5")],
    ids=["literal-na", "empty-str", "str-none", "str-null", "zero", "false", "float", "decimal"],
)
def test_is_missing_false_for_values(value):
    assert is_missing(value) is False


def test_partition_missing_splits_missing_from_present():
    """Missing and present indices keep their original order."""
    missing, present = partition_missing([None, float("nan"), "a", "b"])

    assert missing == [0, 1]
    assert present == [2, 3]


def test_partition_missing_of_values_only():
    """An all-value input has no missing indices."""
    missing, present = partition_missing(["a", "b"])

    assert missing == []
    assert present == [0, 1]


def test_partition_missing_of_missing_only():
    """An all-missing input has no present indices."""
    missing, present = partition_missing([None, None])

    assert missing == [0, 1]
    assert present == []
