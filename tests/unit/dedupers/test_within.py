"""Liken deduper unit tests"""

from datetime import date
from decimal import Decimal

import pyarrow as pa
import pytest

import liken as lk
from liken.exceptions import InvalidArgumentTypeError
from liken.exceptions import InvalidArgumentValueError


#############
# within    #
#############


def test_within_requires_tol():
    """`tol` has no default: a tolerance is data-scale-dependent."""
    with pytest.raises(TypeError):
        lk.within()


@pytest.mark.parametrize(
    "tol",
    [None, "0.5", [1], True],
    ids=["none", "str", "list", "bool"],
)
def test_within_rejects_non_number_tol(tol):
    with pytest.raises(InvalidArgumentTypeError, match="`tol` must be a real number"):
        lk.within(tol=tol)


def test_within_rejects_huge_integer_tol():
    """An int too large for a float is refused at construction, not at
    execution with a raw OverflowError from the window arithmetic."""
    with pytest.raises(InvalidArgumentValueError, match="`tol` must be a finite number"):
        lk.within(tol=10**400)


@pytest.mark.parametrize(
    "tol",
    [-1, -0.5, float("nan"), float("inf"), float("-inf")],
    ids=["negative", "negative-float", "nan", "inf", "neg-inf"],
)
def test_within_rejects_bad_tol_value(tol):
    with pytest.raises(InvalidArgumentValueError):
        lk.within(tol=tol)


@pytest.mark.parametrize(
    "tol",
    [0, 0.5, 100],
    ids=["zero-int", "float", "int"],
)
def test_within_accepts_valid_tol(tol):
    assert lk.within(tol=tol)._tol == tol


@pytest.mark.parametrize(
    "values, tol, expected",
    [
        # close pair within tolerance
        ([10.0, 10.5, 20.0], 0.6, [(0, 1)]),
        # three-way chain: 1.0-1.01 and 1.01-1.02 match, 1.0-1.02 does not
        ([1.0, 1.01, 1.02], 0.01, [(0, 1), (1, 2)]),
        # far apart: nothing matches
        ([1.0, 5.0, 9.0], 1.0, []),
        # unsorted input still pairs correctly
        ([30.0, 10.1, 20.0, 10.0], 0.2, [(1, 3)]),
        # duplicates match each other
        ([7.0, 7.0, 7.0], 0.0, [(0, 1), (0, 2), (1, 2)]),
    ],
)
def test_within_matches_pairs_within_tolerance(values, tol, expected):
    pairs = sorted(lk.within(tol=tol)._gen_similarity_pairs(pa.array(values)))

    assert pairs == expected


@pytest.mark.parametrize(
    "a, b, tol",
    [
        (1.0, 1.01, 0.01),
        (100.0, 100.4, 0.4),
    ],
)
def test_within_matches_inclusively_at_the_bound(a, b, tol):
    pairs = list(lk.within(tol=tol)._gen_similarity_pairs(pa.array([a, b])))

    assert pairs == [(0, 1)]


def test_within_rejects_pair_just_above_bound():
    pairs = list(lk.within(tol=0.01)._gen_similarity_pairs(pa.array([1.0, 1.011])))

    assert pairs == []


def test_within_boundary_guard_admits_rounding_drift():
    """`1.02 - 1.01` computes to 0.010000000000000009, above tol=0.01.

    The guard widens the bound by 1e-9 so the spec's own example matches.
    """
    pairs = list(lk.within(tol=0.01)._gen_similarity_pairs(pa.array([1.01, 1.02])))

    assert pairs == [(0, 1)]


def test_within_zero_tol_is_exact_equality():
    """`tol=0` reduces to numeric equality: only exactly equal values pair.

    No boundary guard applies at tol=0: equal values subtract to exactly
    0.0, so equality is exact, not guarded.
    """
    pairs = sorted(lk.within(tol=0)._gen_similarity_pairs(pa.array([1.0, 1.0, 1.5, 2])))

    assert pairs == [(0, 1)]


@pytest.mark.parametrize(
    "values",
    [[1.0, None, None, 5.0], [1.0, float("nan"), None, 5.0]],
    ids=["none-nulls", "nan-and-null"],
)
def test_within_nulls_pair_with_nulls_only(values):
    """Nulls pair with nulls only; the two values are too far apart to pair."""
    pairs = sorted(lk.within(tol=0.5)._gen_similarity_pairs(pa.array(values)))

    assert pairs == [(1, 2)]


def test_within_literal_zero_is_a_value_not_missing():
    """0.0 is a value: it pairs with 0.0, not with a null."""
    pairs = sorted(lk.within(tol=0.1)._gen_similarity_pairs(pa.array([0.0, None, 0.05])))

    assert pairs == [(0, 2)]


def test_within_empty_and_single_value_arrays_yield_nothing():
    deduper = lk.within(tol=1.0)

    assert list(deduper._gen_similarity_pairs(pa.array([]))) == []
    assert list(deduper._gen_similarity_pairs(pa.array([1.0]))) == []


def test_within_infinite_values_never_match():
    """Two infinities subtract to NaN, which matches nothing."""
    pairs = list(lk.within(tol=1.0)._gen_similarity_pairs(pa.array([float("inf"), float("inf")])))

    assert pairs == []


def test_within_large_integers_survive_float_conversion():
    """Integers beyond float53 precision collapse to the same float and pair.

    Python ints keep full precision; converting to float before comparison
    would merge 2**53 and 2**53 + 1. The pair decision runs on the raw
    values.
    """
    big = 2**53
    pairs = list(lk.within(tol=0)._gen_similarity_pairs(pa.array([big, big + 1], type=pa.int64())))

    assert pairs == []


def test_within_pair_emission_is_deterministic():
    """Repeated runs over the same input emit identical pair lists.

    Multi-pair input: emission order is window-scan order over the sorted
    values, not globally sorted; only run-to-run stability is claimed.
    """
    deduper = lk.within(tol=0.5)

    first = list(deduper._gen_similarity_pairs(pa.array([3.0, 1.2, 1.0, 1.1])))
    second = list(deduper._gen_similarity_pairs(pa.array([3.0, 1.2, 1.0, 1.1])))

    assert first == [(2, 3), (1, 2), (1, 3)]
    assert first == second


@pytest.mark.parametrize(
    "array",
    [
        pa.array(["a", "b"]),
        pa.array([True, False]),
        pa.array([date(2024, 1, 1), date(2024, 1, 2)]),
    ],
    ids=["strings", "booleans", "dates"],
)
def test_within_rejects_non_numeric_arrays_at_execution(array):
    with pytest.raises(InvalidArgumentTypeError, match="numeric"):
        list(lk.within(tol=1.0)._gen_similarity_pairs(array))


@pytest.mark.parametrize(
    "array",
    [
        pa.array([1, 2, 3]),
        pa.array([Decimal("1.5"), Decimal("2.5")]),
        pa.array([1.0, 2.0]),
    ],
    ids=["integers", "decimals", "floats"],
)
def test_within_accepts_numeric_arrays(array):
    assert isinstance(list(lk.within(tol=1.0)._gen_similarity_pairs(array)), list)


def test_within_requires_single_string_column():
    with pytest.raises(InvalidArgumentTypeError):
        lk.within(tol=1.0).validate(("a", "b"))


def test_within_str_renders_tol():
    assert "within(tol=0.5)" in str(lk.within(tol=0.5))
