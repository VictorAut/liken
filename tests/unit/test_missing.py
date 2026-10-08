"""Tests for the shared missing-value helper"""

from decimal import Decimal

import pytest

from liken.core.missing import is_missing


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
