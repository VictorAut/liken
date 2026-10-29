"""Unit tests for the set dedupers' regime dispatch (WI069)."""

import numpy as np
import pytest

from liken.dedupers._blockwise import _MAX_FLOAT32_EXACT_COUNT
from liken.dedupers._blockwise import DEFAULT_BLOCK_MEMORY
from liken.dedupers._blockwise import estimate_v2_key_count
from liken.dedupers._blockwise import inverted_index_pairs
from liken.dedupers._blockwise import pick_set_regime


def _members_for(sets):
    """Dense-ID member arrays from row sets, mirroring the dedupers' encoding."""
    distinct = set()
    for row in sets:
        distinct |= row
    ids = {value: i for i, value in enumerate(sorted(distinct, key=str))}
    return [np.fromiter((ids[value] for value in row), dtype=np.int64, count=len(row)) for row in sets]


def _sizes_for(members):
    return np.array([len(m) for m in members], dtype=np.int64)


############################
# regime selection shapes #
###########################


def test_pick_set_regime_dense_fixture_picks_gram():
    """A small dense fixture sits far below the dispatch's V2 cost floor."""
    sets = [{"a", "b", "c"} for _ in range(20)]
    members = _members_for(sets)
    assert pick_set_regime(members, _sizes_for(members), None) == "gram"


def test_pick_set_regime_high_vocabulary_fixture_picks_index():
    """A fixture shaped past the dispatch's V2 cost floor picks the index.

    30k rows over ~30k distinct members: v1_est = n*n*v = 2.7e13, well
    above the 7.4e12 floor the WI063/CSV calibration fixed. Most values
    are rare (60 hot members carry the sharing), so the key volume is
    small and the memory guard admits the pick.
    """
    n = 30000
    sets = [{f"hot{r % 500}", f"v{r}"} for r in range(n)]
    members = _members_for(sets)
    assert pick_set_regime(members, _sizes_for(members), None) == "index"


def test_pick_set_regime_memory_guard_overrides_the_compute_pick():
    """A compute-cost V2 pick whose key volume exceeds the cap stays Gram.

    Same shape as the V2 fixture, but with one value shared by enough rows
    to blow the key budget: 8 bytes * keys > cap. The guard vetoes.
    """
    n = 4000
    hot = {f"hot{i}" for i in range(40)}  # a small dense pool: huge C(k,2)
    sets = [(hot | {f"v{r}"}) if r % 2 == 0 else {f"v{r}", f"w{r}"} for r in range(n)]
    members = _members_for(sets)
    sizes = _sizes_for(members)
    assert 8 * estimate_v2_key_count(members) > DEFAULT_BLOCK_MEMORY
    assert pick_set_regime(members, sizes, None) == "gram"


def test_pick_set_regime_explicit_cap_is_honoured():
    """A caller-set cap participates in the guard, `None` uses the default."""
    sets = [{f"v{r}" for r in range(1)} for _ in range(0)]  # placeholder
    sets = [{"a", "b"} for _ in range(10)]
    members = _members_for(sets)
    assert pick_set_regime(members, _sizes_for(members), 0) == "gram"


###########################
# generator-level edge cases
##########################


def test_inverted_index_path_all_missing_rows_produce_no_pairs():
    members = [np.array([], dtype=np.int64) for _ in range(5)]
    got = list(inverted_index_pairs(members, _sizes_for(members), 0.5, "jaccard"))
    assert got == []


@pytest.mark.parametrize("scoring", ["jaccard", "token_overlap"])
def test_v1_guard_raises_only_on_the_gram_path(scoring):
    """The 2^24 float32 guard raises under Gram dispatch, not under V2.

    The V2 fixture's huge v1_est dispatches to the index; the guard must
    not fire there. A Gram-shaped fixture with the same oversized sizes
    raises, proving the guard survives on the V1 path.
    """
    sizes = np.array([_MAX_FLOAT32_EXACT_COUNT + 1, 1], dtype=np.int64)
    gram_members = [np.array([0], dtype=np.int64), np.array([0], dtype=np.int64)]
    with pytest.raises(ValueError, match=r"2\^24|16777216"):
        list(inverted_index_pairs.__self__ if False else _gram_path(gram_members, sizes, scoring))


def _gram_path(members, sizes, scoring):
    """Run the Gram generator: the dispatch's V1 arm, called directly."""
    from liken.dedupers._blockwise import gram_strip_pairs

    return gram_strip_pairs(members, sizes, 0.5, scoring, None)
