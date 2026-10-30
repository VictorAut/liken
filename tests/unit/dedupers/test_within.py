"""Liken deduper unit tests"""

from datetime import date
from decimal import Decimal

import pyarrow as pa
import pytest

import liken as lk
from liken.core.disjoint_set import DisjointSet
from liken.exceptions import InvalidArgumentTypeError
from liken.exceptions import InvalidArgumentValueError


def _components(n: int, pairs) -> frozenset[frozenset[int]]:
    """Partition of 0..n-1 implied by a pair stream, as a frozenset of
    frozensets.

    Identity-independent: union-find root labels depend on merge order,
    component membership does not.
    """
    ds = DisjointSet(n)
    for i, j in pairs:
        ds.union(i, j)
    root: dict[int, int] = {i: ds[i] for i in range(n)}
    return frozenset(frozenset(i for i in range(n) if root[i] == r) for r in set(root.values()))


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
        # duplicates merge into one group
        ([7.0, 7.0, 7.0], 0.0, [(0, 1), (0, 2), (1, 2)]),
    ],
)
def test_within_matches_pairs_within_tolerance(values, tol, expected):
    """The partition implied by the pair stream matches the expected groups."""
    n = len(values)
    pairs = lk.within(tol=tol)._gen_similarity_pairs(pa.array(values))

    assert _components(n, pairs) == _components(n, expected)


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
    """`tol=0` reduces to numeric equality: only exactly equal values group.

    No boundary guard applies at tol=0: equal values subtract to exactly
    0.0, so equality is exact, not guarded.
    """
    pairs = list(lk.within(tol=0)._gen_similarity_pairs(pa.array([1.0, 1.0, 1.5, 2])))

    assert _components(4, pairs) == _components(4, [(0, 1)])


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
    """Repeated runs over the same input emit an identical pair stream.

    Only run-to-run stability is claimed; the grouping is asserted
    elsewhere, the emission order is not part of the contract.
    """
    deduper = lk.within(tol=0.5)
    values = pa.array([3.0, 1.2, 1.0, 1.1])

    first = list(deduper._gen_similarity_pairs(values))
    second = list(deduper._gen_similarity_pairs(values))

    assert first == second
    assert _components(4, first) == _components(4, [(2, 3), (1, 2), (1, 3)])


def test_within_pair_stream_is_linear_in_dense_regions():
    """The pair stream is bounded by n-1 value edges plus missing edges.

    A dense region (here: identical values) is where the window scan would
    yield quadratic all-pairs output; the chain keeps it linear.
    """
    pairs = list(lk.within(tol=0)._gen_similarity_pairs(pa.array([7.0, 7.0, 7.0, 7.0, 7.0])))

    assert len(pairs) == 4


@pytest.mark.parametrize(
    "values, tol, null_at",
    [
        ([10.0, 10.4, 10.9, 11.3, 12.0, 20.0, 5.0, 5.05], 0.5, []),
        ([3.0, 3.001, 3.002, 3.004, 2.999], 0.0, []),
        ([1.0, 1.01, 1.02, 1.03, 1.05, 2.0], 0.01, []),
        ([-4.5, -4.0, 0.0, 4.0, 4.5, 4.6], 1.1, []),
        ([0.0, 0.0, 1.0, None, None, 1.5], 0.6, [3, 4]),
        ([2.0, None, None, None, 2.1, 9.0], 0.5, [1, 2, 3]),
        ([5.0, 5.2, 5.21, 5.3], 0.11, []),
        ([1.0, 2.0, 3.0], 0.0, []),
    ],
    ids=[
        "spread",
        "tight-chain-zero-tol",
        "boundary-chain",
        "wide-tol-bridged",
        "nulls-among-values",
        "null-block",
        "guard-drift",
        "separated",
    ],
)
def test_within_partition_matches_all_pairs_reference(values, tol, null_at):
    """Chain edges must imply the same partition as all within-bound pairs.

    Single-linkage equivalence: within any window the adjacent sorted pair
    minimises the difference and the differences telescope, so a window
    holds a within-bound pair only through within-bound adjacent steps;
    transitivity then covers every other pair. Reference: brute-force
    all-pairs bound check.
    """
    values = list(values)
    for i in null_at:
        values[i] = None

    n = len(values)
    deduper = lk.within(tol=tol)
    actual = _components(n, deduper._gen_similarity_pairs(pa.array(values)))

    # The reference mirrors the documented bound, guard included: a
    # difference rounding to at most tol + 1e-9 matches (tol > 0). Missing
    # detection mirrors `is_missing`: None or not equal to itself (NaN).
    guard = 1e-9 if tol > 0 else 0.0
    missing = [i for i, v in enumerate(values) if v is None or v != v]
    present = [i for i, v in enumerate(values) if not (v is None or v != v)]
    reference_pairs = [
        (i, j) for a, i in enumerate(present) for j in present[a + 1 :] if abs(values[i] - values[j]) <= tol + guard
    ]
    reference_pairs.extend((missing[0], m) for m in missing[1:])
    expected = _components(n, reference_pairs)

    assert actual == expected


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
