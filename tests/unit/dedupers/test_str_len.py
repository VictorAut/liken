"""Liken deduper unit tests"""

import pyarrow as pa
import pytest

import liken as lk
from liken.exceptions import InvalidArgumentTypeError
from liken.exceptions import InvalidArgumentValueError


@pytest.mark.parametrize("kwargs", [{"min_len": -1}, {"max_len": -1}])
def test_str_len_rejects_negative_bounds(kwargs):
    """A negative length bound is refused at construction; it behaves as 0."""
    with pytest.raises(InvalidArgumentValueError, match="must be an integer in the range \\[0, inf\\), got -1"):
        lk.str_len(**kwargs)


################################
# eager column validation     #
###############################

##############
#  str_len  #
##############


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_len": True},
        {"max_len": True},
        {"min_len": False},
        {"min_len": "3"},
        {"max_len": "3"},
        {"min_len": 3.0},
        {"max_len": 3.0},
        {"min_len": None},
    ],
    ids=["min-bool", "max-bool", "min-false", "min-str", "max-str", "min-float", "max-float", "min-none"],
)
def test_str_len_rejects_non_integer_bounds(kwargs):
    """A bound that is not an integer — bool, str, float, `None` — is refused at construction."""
    with pytest.raises(InvalidArgumentTypeError, match="must be an integer, got"):
        lk.str_len(**kwargs)


def test_str_len_rejects_inverted_bounds():
    """An empty interval is refused at construction, like num_range."""
    with pytest.raises(ValueError):
        lk.str_len(min_len=10, max_len=9)


def test_str_len_allows_int_bounds_and_none_upper():
    """An int bound is valid; `max_len` may be `None`; `min_len == max_len` is legal."""
    lk.str_len(min_len=3, max_len=None)
    lk.str_len(min_len=3, max_len=3)


@pytest.mark.parametrize(
    "kwargs, values, expected_pairs",
    [
        # band: lengths 3, 4 and 5 sit inside; pairs are star-shaped from
        # the first matched index (predicate semantics: one collapsed group)
        ({"min_len": 3, "max_len": 5}, ["ab", "abc", "abcd", "abcde", "abcdef"], [(1, 2), (1, 3)]),
        # inclusive endpoints: lengths 3 and 5 sit at the bounds and must match
        ({"min_len": 3, "max_len": 5}, ["abc", "abcd", "abcde"], [(0, 1), (0, 2)]),
        # min-only: everything of length >= 2
        ({"min_len": 2}, ["a", "ab", "abc"], [(1, 2)]),
        # max-only: everything of length <= 2
        ({"max_len": 2}, ["a", "ab", "abc"], [(0, 1)]),
    ],
    ids=["band", "inclusive-endpoints", "min-only", "max-only"],
)
def test_str_len_matches_lengths_inside_inclusive_bounds(kwargs, values, expected_pairs):
    """A value matches iff min_len <= len(v) <= max_len, each bound applied only when given."""
    deduper = lk.str_len(**kwargs)

    pairs = sorted(deduper._gen_similarity_pairs(pa.array(values)))

    assert pairs == expected_pairs


def test_str_len_min_equals_max_matches_exact_length():
    """min_len == max_len is the exact-length idiom, symmetric with num_range's one-value interval."""
    deduper = lk.str_len(min_len=3, max_len=3)

    pairs = sorted(deduper._gen_similarity_pairs(pa.array(["abc", "xyz", "abcd", "ab"])))

    assert pairs == [(0, 1)]


def test_str_len_empty_string_never_matches_positive():
    """The empty string is excluded whatever the bounds, even at min_len=0."""
    deduper = lk.str_len(min_len=0)

    assert list(deduper._gen_similarity_pairs(pa.array(["", "ab"]))) == []


def test_str_len_empty_string_matches_only_under_negation():
    """Under ~str_len the empty string negate-matches, like every non-matching value."""
    deduper = ~lk.str_len(min_len=0)

    assert list(deduper._gen_similarity_pairs(pa.array(["", "", "ab"]))) == [(0, 1)]


def test_str_len_mask_is_null_at_missing_positions():
    """Missing positions are null in the mask, not False.

    An explicitly nulled mask survives `pc.invert` on the negated path,
    whatever the base mask's missing-value behaviour evolves into.
    """
    deduper = lk.str_len(min_len=1, max_len=3)

    mask = deduper._vectorized_matches(pa.array(["ab", None]))

    assert mask.is_valid().to_pylist() == [True, False]


def test_str_len_null_never_matches_positive():
    """On the positive path, a missing value matches nothing."""
    deduper = lk.str_len(min_len=1, max_len=3)

    pairs = list(deduper._gen_similarity_pairs(pa.array(["ab", None, "abc"])))

    assert pairs == [(0, 2)]


def test_str_len_null_never_matches_negated():
    """On the negated path, a missing value matches nothing."""
    deduper = ~lk.str_len(min_len=1, max_len=3)

    pairs = list(deduper._gen_similarity_pairs(pa.array(["abcd", None, ""])))

    assert pairs == [(0, 2)]


def test_str_len_fallback_never_matches_missing_or_empty():
    """The Python fallback path: a missing value or the empty string never matches."""
    deduper = lk.str_len(min_len=3, max_len=5)

    assert deduper._matches(None) is False
    assert deduper._matches("") is False
    assert deduper._matches("abc") is True
    assert deduper._matches("ab") is False
    assert deduper._matches("abcdef") is False


def test_str_len_negated_fallback_never_matches_missing():
    """The negated Python fallback also never matches a missing value; the empty string does."""
    deduper = ~lk.str_len(min_len=3, max_len=5)

    assert deduper._matches(None) is False
    assert deduper._matches("") is True
    assert deduper._matches("abc") is False
    assert deduper._matches("ab") is True
