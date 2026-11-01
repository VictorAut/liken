"""Unit tests for the set dedupers' regime dispatch (WI069)."""

import logging

import numpy as np
import pyarrow as pa
import pytest

import liken as lk
from liken.dedupers._blockwise import _MAX_FLOAT32_EXACT_COUNT
from liken.dedupers._blockwise import _V2_KEY_BUDGET
from liken.dedupers._blockwise import _V2_MIN_V1_COST
from liken.dedupers._blockwise import DEFAULT_BLOCK_MEMORY
from liken.dedupers._blockwise import estimate_v2_key_count
from liken.dedupers._blockwise import inverted_index_pairs
from liken.dedupers._blockwise import pick_set_regime
from liken.exceptions import InvalidArgumentTypeError
from liken.exceptions import InvalidArgumentValueError


def _members_for(sets):
    """Dense-ID member arrays from row sets, mirroring the dedupers' encoding."""
    distinct = set()
    for row in sets:
        distinct |= row
    ids = {value: i for i, value in enumerate(sorted(distinct, key=str))}
    return [np.fromiter((ids[value] for value in row), dtype=np.int64, count=len(row)) for row in sets]


def _sizes_for(members):
    return np.array([len(m) for m in members], dtype=np.int64)


def _table_for(sets):
    """A compound table whose ingested row sets equal `sets`.

    One column per member slot of the widest set; each row's members fill
    its columns in sorted order and the rest stay None (missing), so the
    deduper's ingestion reproduces each set exactly.
    """
    width = max((len(s) for s in sets), default=1)
    columns = [[] for _ in range(width)]
    for row in sets:
        values = sorted(row, key=str)
        for c in range(width):
            columns[c].append(values[c] if c < len(values) else None)
    return pa.table({f"c{c}": pa.array(columns[c]) for c in range(width)})


############################
# regime selection shapes #
###########################


def test_pick_set_regime_dense_fixture_picks_gram():
    """A small dense fixture sits far below the dispatch's V2 cost floor."""
    sets = [{"a", "b", "c"} for _ in range(20)]
    members = _members_for(sets)
    assert pick_set_regime(members, "auto") == "gram"


def test_pick_set_regime_high_vocabulary_fixture_picks_index():
    """A fixture shaped past the dispatch's V2 cost floor picks the index.

    30k rows over ~30k distinct members: v1_est = n*n*v = 2.7e13, above
    the 3.1e11 floor the corrected WI069 calibration fixed. Most values
    are rare (500 hot members carry the sharing), so the key volume is
    small and the memory guard admits the pick.
    """
    n = 30000
    sets = [{f"hot{r % 500}", f"v{r}"} for r in range(n)]
    members = _members_for(sets)
    assert pick_set_regime(members, "auto") == "index"


def test_pick_set_regime_memory_guard_overrides_the_compute_pick():
    """A compute-cost V2 pick whose key volume exceeds the guard budget stays Gram.

    One value shared by 8500 rows emits C(8500, 2) = 36.1M keys; 8 bytes
    per key is 275 MiB, over the 256 MiB `_V2_KEY_BUDGET`. Each row also
    holds a unique member, so v1_est = 8500**2 * 8501 = 6.1e11 is past
    the compute floor: without the guard the compute pick would be index.
    """
    n = 8500
    sets = [{"hot", f"v{r}"} for r in range(n)]
    members = _members_for(sets)
    keys = estimate_v2_key_count(members)
    assert _V2_MIN_V1_COST < 8500 * 8500 * 8501
    assert 8 * keys > _V2_KEY_BUDGET
    assert pick_set_regime(members, "auto") == "gram"


def test_pick_set_regime_explicit_strategies_bypass_the_guard_and_the_floor():
    """`"gram"` / `"index"` are honored on any shape; only `"auto"` computes."""
    sets = [{"hot", f"v{r}"} for r in range(8500)]
    members = _members_for(sets)
    assert pick_set_regime(members, "index") == "index"
    assert pick_set_regime(members, "gram") == "gram"


def test_v2_key_budget_is_four_times_the_default_block_cap():
    assert _V2_KEY_BUDGET == 4 * DEFAULT_BLOCK_MEMORY


############################
# deduper-level wiring    #
##########################


@pytest.mark.parametrize("factory", [lk.jaccard, lk.token_overlap])
def test_strategy_defaults_to_auto(factory):
    assert factory()._strategy == "auto"


@pytest.mark.parametrize("factory", [lk.jaccard, lk.token_overlap])
def test_strategy_rejects_invalid_values(factory):
    with pytest.raises(InvalidArgumentValueError, match="strategy"):
        factory(strategy="bogus")
    with pytest.raises(InvalidArgumentTypeError, match="strategy"):
        factory(strategy=1)


@pytest.mark.parametrize("factory", [lk.jaccard, lk.token_overlap])
def test_forced_strategy_changes_the_picked_regime(factory, caplog):
    """The deduper routes `strategy` into the selector; the log shows the pick."""
    sets = [{"a", "b"} for _ in range(10)]
    table = _table_for(sets)
    with caplog.at_level(logging.DEBUG, logger="liken.dedupers._blockwise"):
        list(factory()._gen_similarity_pairs(table))
        assert "regime=gram" in caplog.text
        caplog.clear()
        list(factory(strategy="index")._gen_similarity_pairs(table))
        assert "regime=index" in caplog.text


@pytest.mark.parametrize("factory", [lk.jaccard, lk.token_overlap])
def test_forced_regimes_agree_with_the_auto_pick(factory):
    """On a fixture auto sends to gram, a forced index run is byte-identical.

    The three-way rotation makes several duplicate groups, so the output
    is non-trivial.
    """
    sets = [{"a", "b"}, {"b", "c"}, {"a", "c"}] * 10
    table = _table_for(sets)
    auto = list(factory()._gen_similarity_pairs(table))
    forced = list(factory(strategy="index")._gen_similarity_pairs(table))
    assert auto == forced
    assert len(auto) == 3 * 45  # each duplicate group of 10 rows emits C(10, 2)


def test_gram_forcing_overrides_an_auto_index_pick(caplog):
    """`strategy="gram"` forces V1 on a fixture `auto` sends to index.

    1600 rows of ~95 mostly-unique members put v1_est (n*n*v = 3.9e11)
    past the 3.1e11 floor with a tiny key volume (one shared member plus
    a 4-way rotation), so auto picks the index; an explicit `"gram"` must
    still run the Gram generator to completion.
    """
    n = 1600
    sets = [{f"v{r}_{i}" for i in range(95)} | {"hot", f"p{r % 4}"} for r in range(n)]
    table = _table_for(sets)
    with caplog.at_level(logging.DEBUG, logger="liken.dedupers._blockwise"):
        list(lk.jaccard()._gen_similarity_pairs(table))
        assert "regime=index" in caplog.text
        caplog.clear()
        list(lk.jaccard(strategy="gram")._gen_similarity_pairs(table))
        assert "regime=gram" in caplog.text


@pytest.mark.parametrize("factory", [lk.jaccard, lk.token_overlap])
@pytest.mark.parametrize("strategy", ["auto", "gram", "index"])
def test_degenerate_inputs_produce_no_pairs_under_every_strategy(factory, strategy):
    assert list(factory(strategy=strategy)._gen_similarity_pairs(pa.table({"c0": pa.array([])}))) == []
    table = _table_for([{f"v{r}"} for r in range(1)])
    assert list(factory(strategy=strategy)._gen_similarity_pairs(table)) == []
    table = _table_for([set() for _ in range(5)])
    assert list(factory(strategy=strategy)._gen_similarity_pairs(table)) == []


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

    A Gram-shaped fixture with oversized sizes raises through
    `gram_strip_pairs`; the V2 path (int64 counts) raises nothing.
    """
    from liken.dedupers._blockwise import gram_strip_pairs

    sizes = np.array([_MAX_FLOAT32_EXACT_COUNT + 1, 1], dtype=np.int64)
    members = [np.array([0], dtype=np.int64), np.array([0], dtype=np.int64)]
    with pytest.raises(ValueError, match=r"2\^24|16777216"):
        list(gram_strip_pairs(members, sizes, 0.5, scoring, None))
    # V2's int64 counts carry no such bound and raise nothing. The pairs
    # depend on the scoring: containment of the 1-member row in the huge
    # row is 1.0 and matches; jaccard is ~2^-24 and does not.
    assert list(inverted_index_pairs(members, sizes, 0.5, scoring, None)) == (
        [(0, 1)] if scoring == "token_overlap" else []
    )
