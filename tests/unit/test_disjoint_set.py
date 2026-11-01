"""Unit tests for the internal DisjointSet union-find structure."""

import pytest

from liken.core.disjoint_set import DisjointSet


################
# construction #
################


def test_constructs_with_zero_elements():
    ds = DisjointSet(0)
    assert len(ds) == 0


def test_constructs_with_single_element():
    ds = DisjointSet(1)
    assert ds[0] == 0


@pytest.mark.parametrize("n", [-1, -5], ids=["minus-one", "minus-five"])
def test_rejects_negative_n(n):
    with pytest.raises(ValueError, match="non-negative"):
        DisjointSet(n)


########
# union #
########


def test_union_makes_two_elements_share_root():
    ds = DisjointSet(2)
    ds.union(0, 1)
    assert ds[0] == ds[1]


def test_union_is_no_op_when_already_merged():
    ds = DisjointSet(3)
    ds.union(0, 1)
    root_before = ds[1]
    ds.union(1, 0)  # reversed order, same pair
    assert ds[1] == root_before
    assert ds[0] == root_before
    assert ds[2] != ds[0]  # unrelated element stays separate


def test_chain_merges_collapse_to_one_root():
    """0-1, 1-2, 2-3 ... in a chain: every member must report the same root."""
    ds = DisjointSet(10)
    for i in range(9):
        ds.union(i, i + 1)
    roots = {ds[i] for i in range(10)}
    assert len(roots) == 1


def test_union_by_size_keeps_larger_set_root():
    """The root of the larger set wins; a singleton joining a pair takes the pair's root."""
    ds = DisjointSet(3)
    ds.union(0, 1)  # {0, 1}, root 0
    ds.union(2, 0)  # 2 joins the larger set, so root must stay 0
    assert ds[0] == ds[1] == ds[2]


def test_roots_stable_after_multiple_merges():
    ds = DisjointSet(6)
    ds.union(0, 1)
    ds.union(2, 3)
    first_roots = (ds[0], ds[2])
    ds.union(4, 5)
    ds.union(0, 5)
    # merging {0,1} with {4,5} must not disturb {2,3}'s root
    assert ds[2] == first_roots[1]
    assert ds[0] == ds[1] == ds[4] == ds[5]


###################
# partition shape #
###################


def test_partition_round_trip_after_random_merges():
    """After a merge stream, ds[i] must be constant on each connected
    component and distinct across them."""
    pairs = [(0, 1), (2, 3), (4, 5), (1, 3), (5, 0), (2, 5)]
    ds = DisjointSet(7)
    for i, j in pairs:
        ds.union(i, j)

    # 0-1, 2-3, 4-5 chained through 1-3, 5-0 -> all of 0..5 merge;
    # 6 never appears so stays alone.
    assert ds[6] not in {ds[i] for i in range(6)}
    assert len({ds[i] for i in range(7)}) == 2


#######################################
# brute-force reference cross-check #
#######################################


def _naive_components(n: int, pairs: list[tuple[int, int]]) -> list[frozenset[int]]:
    """Reference partition: per-pair union of index sets, then the
    component of each element as a frozenset (identity-independent)."""
    sets: list[set[int]] = [{i} for i in range(n)]
    for i, j in pairs:
        a, b = sets[i], sets[j]
        if a is not b:
            a |= b
            for x in b:
                sets[x] = a
    # All members of a component now share one set object.
    return [frozenset(s) for s in {id(s): s for s in sets}.values()]


def test_matches_brute_force_reference_over_random_streams():
    """DisjointSet's partition must match a naive per-pair union over
    many random pair streams (fixed seeds, black-box comparison via
    component membership, not root labels)."""
    import random

    n = 40
    for seed in range(20):
        rng = random.Random(seed)
        pairs = [(rng.randrange(n), rng.randrange(n)) for _ in range(120)]

        ds = DisjointSet(n)
        for i, j in pairs:
            ds.union(i, j)

        expected = _naive_components(n, pairs)

        # Same-partition check: every pair of indices shares a root iff
        # the reference puts them in the same component.
        ref_index: dict[int, frozenset[int]] = {}
        for component in expected:
            for member in component:
                ref_index[member] = component
        for a in range(n):
            for b in range(n):
                same_ref = ref_index[a] == ref_index[b]
                same_ds = ds[a] == ds[b]
                assert same_ds == same_ref, f"seed {seed}: mismatch on ({a}, {b})"
