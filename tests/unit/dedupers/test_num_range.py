"""Liken deduper unit tests"""

import pyarrow as pa
import pytest

import liken as lk
from liken.exceptions import InvalidArgumentTypeError
from liken.exceptions import InvalidArgumentValueError


##############
# num_range #
##############


def test_num_range_rejects_both_bounds_none():
    """A fully open range matches every non-missing value; refused at construction."""
    with pytest.raises(ValueError):
        lk.num_range(min=None, max=None)


def test_num_range_rejects_min_above_max():
    """An empty interval is refused at construction, like both-None."""
    with pytest.raises(ValueError):
        lk.num_range(min=40_000, max=30_000)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min": True},
        {"max": True},
        {"min": False},
        {"min": "500"},
        {"max": "500"},
    ],
    ids=["min-bool", "max-bool", "min-false", "min-str", "max-str"],
)
def test_num_range_rejects_non_number_bounds(kwargs):
    """A bound that is not a real number — bool, str — is refused at construction."""
    with pytest.raises(InvalidArgumentTypeError, match="must be a real number, got"):
        lk.num_range(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min": float("nan")},
        {"max": float("nan")},
        {"min": float("nan"), "max": 5},
    ],
    ids=["min-nan", "max-nan", "min-nan-with-max"],
)
def test_num_range_rejects_nan_bounds(kwargs):
    """A NaN bound is refused at construction as an invalid value."""
    with pytest.raises(InvalidArgumentValueError, match="must be a real number, got NaN"):
        lk.num_range(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min": 1, "max": 2},
        {"min": 1.5},
        {"max": 1.5},
        {"min": None, "max": 2},
        {"min": 1, "max": None},
    ],
    ids=["ints", "min-float", "max-float", "min-none", "max-none"],
)
def test_num_range_allows_real_or_none_bounds(kwargs):
    """An int or float bound is a real number; `None` means unbounded."""
    lk.num_range(**kwargs)


def test_num_range_min_equals_max_is_allowed():
    """A one-value interval is a valid, non-empty range."""
    lk.num_range(min=5, max=5)


@pytest.mark.parametrize(
    "kwargs, values, expected_pairs",
    [
        # band: only 3, 4 and 5 sit inside; pairs are star-shaped from the
        # first matched index (predicate semantics: one collapsed group)
        ({"min": 3, "max": 5}, [1, 2, 3, 4, 5, 6], [(2, 3), (2, 4)]),
        # inclusive endpoints: 1 and 5 sit at the bounds and must match
        ({"min": 1, "max": 5}, [1, 2, 5], [(0, 1), (0, 2)]),
        # min-only: everything >= 30_000
        ({"min": 30_000}, [29_999, 30_000, 30_001], [(1, 2)]),
        # max-only: everything <= 40_000
        ({"max": 40_000}, [39_999, 40_000, 40_001], [(0, 1)]),
    ],
    ids=["band", "inclusive-endpoints", "min-only", "max-only"],
)
def test_num_range_matches_values_inside_inclusive_bounds(kwargs, values, expected_pairs):
    """A value matches iff min <= v <= max, each bound applied only when given."""
    deduper = lk.num_range(**kwargs)

    pairs = sorted(deduper._gen_similarity_pairs(pa.array(values)))

    assert pairs == expected_pairs


def test_num_range_empty_match_set_yields_no_pairs():
    """No value in range: no pairs, no crash."""
    deduper = lk.num_range(min=100, max=200)

    assert list(deduper._gen_similarity_pairs(pa.array([1, 2, 3]))) == []


def test_num_range_null_and_nan_never_match_positive():
    """On the positive path, a missing value matches nothing."""
    deduper = lk.num_range(min=1, max=5)

    pairs = list(deduper._gen_similarity_pairs(pa.array([1.0, None, float("nan"), 5.0])))

    assert pairs == [(0, 3)]


def test_num_range_null_and_nan_never_match_negated():
    """On the negated path, a missing value matches nothing.

    The Arrow comparison kernels return False at a float NaN, so a bare
    `pc.invert` of the positive mask would group NaN rows. The deduper's
    mask must null out missing positions before the inversion.
    """
    deduper = ~lk.num_range(min=1, max=5)

    pairs = list(deduper._gen_similarity_pairs(pa.array([0.5, None, float("nan"), 9.0])))

    assert pairs == [(0, 3)]


def test_num_range_fallback_never_matches_missing():
    """The Python fallback path also never matches a missing value."""
    deduper = lk.num_range(min=1, max=5)

    assert deduper._matches(None) is False
    assert deduper._matches(float("nan")) is False
    assert deduper._matches(3) is True


def test_num_range_negated_fallback_never_matches_missing():
    """The negated Python fallback also never matches a missing value."""
    deduper = ~lk.num_range(min=1, max=5)

    assert deduper._matches(None) is False
    assert deduper._matches(float("nan")) is False
    assert deduper._matches(3) is False
    assert deduper._matches(50) is True


def test_num_range_requires_single_string_column():
    with pytest.raises(InvalidArgumentTypeError):
        lk.num_range(min=1).validate(("a", "b"))


def test_num_range_str_renders_name_and_bounds():
    representation = str(lk.num_range(min=1, max=5))

    assert representation == "num_range(min=1, max=5)"
