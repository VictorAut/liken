import math

import numpy as np
import pytest

from liken.core.deduper import BaseDeduper
from liken.core.deduper import ThresholdDeduper
from liken.dedupers.exact import exact
from liken.dedupers.fuzzy import fuzzy
from liken.dedupers.jaccard import jaccard
from liken.exceptions import InvalidArgumentError
from liken.exceptions import InvalidArgumentTypeError
from liken.exceptions import InvalidArgumentValueError
from liken.validations import require_choice
from liken.validations import require_int
from liken.validations import require_non_empty_str
from liken.validations import require_real_number
from liken.validations import require_type
from liken.validations import validate_columns_arg
from liken.validations import validate_explore_column_exists
from liken.validations import validate_explore_columns_arg
from liken.validations import validate_keep_arg
from liken.validations import validate_thresholds_arg


# require_type


def test_require_type_returns_value_on_match():
    assert require_type("columns", "email", str) == "email"


def test_require_type_rejects_wrong_type():
    with pytest.raises(InvalidArgumentTypeError, match="`columns` must be of type 'str', got 'tuple'"):
        require_type("columns", ("a",), str)


def test_require_type_rejects_wrong_type_via_builtin_and_base():
    with pytest.raises(TypeError):
        require_type("columns", 1, str)
    with pytest.raises(InvalidArgumentError):
        require_type("columns", 1, str)


def test_require_type_accepts_type_tuple():
    assert require_type("key", ("a",), (str, tuple)) == ("a",)
    with pytest.raises(InvalidArgumentTypeError, match="`key` must be of type 'str' or 'tuple', got 'int'"):
        require_type("key", 1, (str, tuple))


@pytest.mark.parametrize("value", [True, np.float64(1.0)])
def test_require_type_rejects_bool_and_numpy_scalar_for_str(value):
    with pytest.raises(InvalidArgumentTypeError):
        require_type("columns", value, str)


# require_real_number


@pytest.mark.parametrize("value", [0, 0.5, 1, -2.5])
def test_require_real_number_accepts_int_and_float(value):
    assert require_real_number("frac", value) == value


def test_require_real_number_accepts_numpy_float64():
    """`np.float64` is a `float` subclass, so the strict builtin check passes it."""
    assert require_real_number("frac", np.float64(0.5)) == 0.5
    assert require_real_number("threshold", np.float64(0.9), low=0, high=1) == 0.9


def test_require_int_rejects_numpy_int64():
    with pytest.raises(InvalidArgumentTypeError, match="`max_distance` must be an integer, got 'int64'"):
        require_int("max_distance", np.int64(2))


@pytest.mark.parametrize("value", [True, "0.5", None, object()])
def test_require_real_number_rejects_non_numbers(value):
    with pytest.raises(InvalidArgumentTypeError, match="`frac` must be a real number, got"):
        require_real_number("frac", value)


def test_require_real_number_rejects_nan():
    with pytest.raises(InvalidArgumentValueError, match="`frac` must be a real number, got NaN"):
        require_real_number("frac", math.nan)


def test_require_real_number_allows_inf_when_unbounded():
    assert require_real_number("x", math.inf) == math.inf


def test_require_real_number_bounds_inclusive_low_exclusive_high():
    assert require_real_number("threshold", 0) == 0
    with pytest.raises(
        InvalidArgumentValueError,
        match="`threshold` must be a real number in the range \\[0, 1\\), got 1",
    ):
        require_real_number("threshold", 1, low=0, high=1)


def test_require_real_number_bounds_exclusive_low_inclusive_high():
    assert require_real_number("frac", 1, low=0, high=1, low_inclusive=False, high_inclusive=True) == 1
    with pytest.raises(InvalidArgumentValueError, match="must be a real number in the range \\(0, 1\\], got 0"):
        require_real_number("frac", 0, low=0, high=1, low_inclusive=False, high_inclusive=True)


def test_require_real_number_bounds_low_only():
    assert require_real_number("x", 0, low=0) == 0
    with pytest.raises(InvalidArgumentValueError, match="must be a real number in the range \\[0, inf\\), got -1"):
        require_real_number("x", -1, low=0)


def test_require_real_number_bounds_high_only():
    with pytest.raises(InvalidArgumentValueError, match="must be a real number in the range \\(-inf, 1\\), got 1"):
        require_real_number("x", 1, high=1)


def test_require_real_number_rejects_bool_against_bounds():
    with pytest.raises(InvalidArgumentTypeError):
        require_real_number("threshold", False, low=0, high=1)


def test_require_real_number_allow_none():
    assert require_real_number("min", None, allow_none=True) is None
    with pytest.raises(InvalidArgumentTypeError):
        require_real_number("min", None, allow_none=False)


# require_int


@pytest.mark.parametrize("value", [0, 3, -1])
def test_require_int_accepts_int(value):
    assert require_int("max_distance", value) == value


@pytest.mark.parametrize("value", [True, 2.0, "2", None, np.int64(2)])
def test_require_int_rejects_non_int(value):
    with pytest.raises(InvalidArgumentTypeError, match="`max_distance` must be an integer, got"):
        require_int("max_distance", value)


def test_require_int_bounds():
    assert require_int("max_distance", 0, low=0) == 0
    with pytest.raises(
        InvalidArgumentValueError, match="`max_distance` must be an integer in the range \\[0, inf\\), got -1"
    ):
        require_int("max_distance", -1, low=0)


def test_require_int_allow_none():
    assert require_int("max_len", None, allow_none=True) is None
    with pytest.raises(InvalidArgumentTypeError):
        require_int("max_len", None, allow_none=False)


# require_non_empty_str


def test_require_non_empty_str_returns_value_on_match():
    assert require_non_empty_str("id", "row_id") == "row_id"


@pytest.mark.parametrize("value", [1, None, ("a",), object()])
def test_require_non_empty_str_rejects_non_str(value):
    with pytest.raises(InvalidArgumentTypeError, match="`id` must be of type 'str', got"):
        require_non_empty_str("id", value)


def test_require_non_empty_str_rejects_empty_string():
    with pytest.raises(InvalidArgumentValueError, match="`id` must be a non-empty string, got ''"):
        require_non_empty_str("id", "")


def test_require_non_empty_str_allow_none():
    assert require_non_empty_str("id", None, allow_none=True) is None
    with pytest.raises(InvalidArgumentTypeError):
        require_non_empty_str("id", None, allow_none=False)


def test_require_non_empty_str_rejects_via_builtin_and_base():
    with pytest.raises(TypeError):
        require_non_empty_str("id", 1)
    with pytest.raises(ValueError):
        require_non_empty_str("id", "")


# require_choice


def test_require_choice_returns_value_on_match():
    assert require_choice("keep", "first", ("first", "last")) == "first"


def test_require_choice_rejects_unknown_value():
    with pytest.raises(
        InvalidArgumentValueError,
        match="`keep` must be one of \\('first', 'last'\\), got 'middle'",
    ):
        require_choice("keep", "middle", ("first", "last"))


def test_require_choice_rejects_via_builtin_and_base():
    with pytest.raises(ValueError):
        require_choice("keep", "middle", ("first", "last"))
    with pytest.raises(InvalidArgumentError):
        require_choice("keep", "middle", ("first", "last"))


# validate_columns_arg


@pytest.mark.parametrize(
    "columns",
    ["address", ("address", "email"), None],
)
def test_validate_columns_arg_accepts_valid(columns):
    assert validate_columns_arg(columns) == columns


@pytest.mark.parametrize("columns", [123, 0, ["address"], {"address"}, {"address": 1}])
def test_validate_columns_arg_rejects_wrong_type(columns):
    """U11/U12/U33: the type check comes first — a wrong type gets a type
    error, never mode advice."""
    with pytest.raises(InvalidArgumentTypeError, match="`columns` must be of type 'str' or 'tuple', got"):
        validate_columns_arg(columns)


def test_validate_columns_arg_rejects_list_with_use_a_tuple_message():
    """U11: lists are the common mistake; the message says what to use."""
    with pytest.raises(
        InvalidArgumentTypeError,
        match="`columns` must be of type 'str' or 'tuple', got 'list'; use a tuple of column labels",
    ):
        validate_columns_arg(["address"])


@pytest.mark.parametrize("columns", [(0,), ("a", 1), ("a", None)])
def test_validate_columns_arg_rejects_non_str_tuple_members(columns):
    """D4/U47: labels must be str; a non-str label is a type error."""
    with pytest.raises(InvalidArgumentTypeError, match="`columns` members must be of type 'str', got"):
        validate_columns_arg(columns)


def test_validate_columns_arg_rejects_repeated_labels():
    with pytest.raises(
        InvalidArgumentValueError,
        match=r"columns labels cannot be repeated\. Repeated labels: 'a'",
    ):
        validate_columns_arg(("a", "a"))


# validate_thresholds_arg


def test_validate_thresholds_arg_accepts_valid():
    assert validate_thresholds_arg([0.5, 0.9]) == [0.5, 0.9]


def test_validate_thresholds_arg_rejects_non_list():
    with pytest.raises(InvalidArgumentTypeError, match="`thresholds` must be of type 'list', got 'str'"):
        validate_thresholds_arg("0.5")


def test_validate_thresholds_arg_rejects_empty():
    with pytest.raises(InvalidArgumentValueError, match="`thresholds` cannot be empty"):
        validate_thresholds_arg([])


@pytest.mark.parametrize(
    "thresholds, expected_match",
    [
        ([0], "must be a real number in the range \\(0, 1\\), got 0"),
        ([1], "must be a real number in the range \\(0, 1\\), got 1"),
        ([1.5], "must be a real number in the range \\(0, 1\\), got 1.5"),
        ([-0.1], "must be a real number in the range \\(0, 1\\), got -0.1"),
        ([True], "must be a real number, got 'bool'"),
        ([0.5, "0.9"], "`thresholds\\[1\\]` must be a real number, got 'str'"),
        ([math.nan], "must be a real number, got NaN"),
    ],
)
def test_validate_thresholds_arg_rejects_invalid_members(thresholds, expected_match):
    with pytest.raises((InvalidArgumentTypeError, InvalidArgumentValueError), match=expected_match):
        validate_thresholds_arg(thresholds)


# validate_explore_columns_arg


def test_validate_explore_columns_arg_accepts_list():
    assert validate_explore_columns_arg(["address", "email"]) == ["address", "email"]


def test_validate_explore_columns_arg_accepts_dict_of_threshold_dedupers():
    columns = {"address": fuzzy()}
    assert validate_explore_columns_arg(columns) == columns


@pytest.mark.parametrize("columns", [[], ["address", 1]])
def test_validate_explore_columns_arg_rejects_invalid_list(columns):
    with pytest.raises(InvalidArgumentValueError, match="columns for explore must be a non-empty list"):
        validate_explore_columns_arg(columns)


@pytest.mark.parametrize("columns", [{}, {1: fuzzy()}])
def test_validate_explore_columns_arg_rejects_invalid_dict(columns):
    with pytest.raises(InvalidArgumentValueError, match="columns for explore must be a non-empty list"):
        validate_explore_columns_arg(columns)


@pytest.mark.parametrize(
    "columns",
    [
        {"address": exact()},  # predicate deduper, not a threshold deduper
        {"address": jaccard()},  # compound-column deduper
    ],
)
def test_validate_explore_columns_arg_rejects_invalid_deduper(columns):
    with pytest.raises(InvalidArgumentValueError, match="explore dedupers must be single-column similarity"):
        validate_explore_columns_arg(columns)


def test_validate_explore_columns_arg_rejects_other_types():
    with pytest.raises(InvalidArgumentValueError, match="columns for explore must be a non-empty list"):
        validate_explore_columns_arg("address")


# validate_explore_column_exists


def test_validate_explore_column_exists_accepts_present_column():
    assert validate_explore_column_exists("address", ["id", "address"]) == "address"


def test_validate_explore_column_exists_rejects_missing_column():
    with pytest.raises(InvalidArgumentValueError, match="column 'address' not found in the dataframe"):
        validate_explore_column_exists("address", ["id", "email"])


# ThresholdDeduper bounds


@pytest.mark.parametrize("threshold", [0.5, 0.99])
def test_threshold_deduper_accepts_threshold_below_one(threshold):
    assert isinstance(ThresholdDeduper(threshold=threshold), BaseDeduper)


@pytest.mark.parametrize("threshold", [-0.1, 0, 1.0, 1.5])
def test_threshold_deduper_rejects_threshold_outside_bounds(threshold):
    """D2: the similarity threshold range is the open (0, 1) — 0 silently
    collapses whole columns and 1 can never match."""
    with pytest.raises(
        InvalidArgumentValueError,
        match="`threshold` must be a real number in the range \\(0, 1\\), got",
    ):
        ThresholdDeduper(threshold=threshold)


def test_threshold_deduper_rejects_bool_threshold():
    with pytest.raises(
        InvalidArgumentTypeError,
        match="`threshold` must be a real number, got 'bool'",
    ):
        ThresholdDeduper(threshold=False)


def test_threshold_deduper_rejects_string_threshold():
    with pytest.raises(
        InvalidArgumentTypeError,
        match="`threshold` must be a real number, got 'str'",
    ):
        ThresholdDeduper(threshold="high")


def test_validate_keep_arg_rejects_non_str_with_type_error():
    with pytest.raises(TypeError):
        validate_keep_arg(1)


def test_validate_keep_arg_rejects_unknown_str_with_value_error():
    with pytest.raises(ValueError):
        validate_keep_arg("middle")
