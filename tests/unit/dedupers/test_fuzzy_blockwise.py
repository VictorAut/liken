"""Unit tests for `fuzzy`'s block-wise all-pairs path and its knobs."""

import random
from decimal import Decimal

import pyarrow as pa
import pytest
from rapidfuzz import fuzz

import liken as lk
from liken.dedupers.fuzzy import _BOUNDARY_TOLERANCE
from liken.exceptions import InvalidArgumentTypeError
from liken.exceptions import InvalidArgumentValueError


def _brute_force_pairs(values, threshold, scorer):
    """Pure-Python reference: the per-row loop's pair output, in yield order.

    Mirrors fuzzy.py's pre-change all-pairs loop exactly, including the
    missing-value star pairs. Scores go through per-row `process.cdist`
    (float32), not direct scorer calls, so the oracle matches the old
    loop's real output at the boundary.
    """
    from rapidfuzz import process

    from liken.core.missing import partition_missing

    missing, present = partition_missing(values)
    pairs = [(missing[0], i) for i in missing[1:]]

    scored = [values[i] for i in present]
    thr = 100 * threshold

    for pos, s1 in enumerate(scored):
        if pos + 1 >= len(scored):
            break
        scores = process.cdist([s1], scored[pos + 1 :], scorer=scorer)[0]
        for offset, score in enumerate(scores):
            if score >= thr - _BOUNDARY_TOLERANCE:
                pairs.append((present[pos], present[pos + 1 + offset]))

    return pairs


_SEEDED_VALUES = None


def _seeded_values(n, seed=1234):
    """Deterministic value list with near-duplicates and nulls, built once."""
    global _SEEDED_VALUES
    if _SEEDED_VALUES is None:
        rng = random.Random(seed)
        base = ["london", "paris", "berlin", "madrid", "rome", "vienna", "dublin", "oslo"]
        values = []
        for _ in range(n):
            if values and rng.random() < 0.3:
                values.append(base[rng.randrange(len(base))])
            else:
                word = "".join(rng.choice("abcdefgh") for _ in range(rng.randrange(3, 10)))
                values.append(word)
        for i in range(n):
            if rng.random() < 0.1:
                values[i] = None
        _SEEDED_VALUES = values
    return _SEEDED_VALUES


#############################
# byte-identical pair output #
############################


@pytest.mark.parametrize(
    "scorer_name",
    ["simple_ratio", "partial_ratio", "token_sort_ratio", "weighted_ratio", "jaro_winkler"],
)
def test_fuzzy_allpairs_byte_identical_to_reference(scorer_name):
    """The block-wise path reproduces the per-row loop's output exactly.

    `partial_ratio` is asymmetric, so it exercises the query direction.
    """
    values = _seeded_values(60)
    scorer = lk.fuzzy(scorer=scorer_name).get_scorer()

    expected = _brute_force_pairs(values, 0.8, scorer)
    got = list(lk.fuzzy(threshold=0.8, scorer=scorer_name)._gen_similarity_pairs(pa.array(values)))

    assert got == expected


def test_fuzzy_allpairs_blockwise_cap_forces_multiple_blocks():
    """A tiny cap still yields identical output, across several cdist calls."""
    values = _seeded_values(40)
    scorer = lk.fuzzy().get_scorer()

    expected = _brute_force_pairs(values, 0.95, scorer)
    got = list(lk.fuzzy(threshold=0.95, max_block_memory=1)._gen_similarity_pairs(pa.array(values)))

    assert got == expected


def test_fuzzy_allpairs_boundary_scores_match():
    """A pair scoring exactly at the boundary matches through the block path."""
    deduper = lk.fuzzy(threshold=0.75)
    values = ["abcd", "abce", "abcf"]

    assert fuzz.ratio("abcd", "abce") == 75.0

    assert list(deduper._gen_similarity_pairs(pa.array(values))) == [(0, 1), (0, 2), (1, 2)]


def test_fuzzy_allpairs_drifted_threshold_still_matches(monkeypatch):
    """The tolerance survives the block path: the drifted `100 * threshold`
    must not reject a pair sitting exactly on the boundary."""
    threshold = 0.42
    exact_score = float(Decimal(str(threshold)) * 100)
    deduper = lk.fuzzy(threshold=threshold)
    monkeypatch.setattr(deduper, "get_scorer", lambda: lambda s1, s2, **kwargs: exact_score)

    assert list(deduper._gen_similarity_pairs(pa.array(["x", "y"]))) == [(0, 1)]


def test_fuzzy_allpairs_nulls_and_missing_shapes():
    """Null star pairs, null-value separation and NaN handling are unchanged."""
    deduper = lk.fuzzy(threshold=0.5)

    array = pa.array(["nathan", None, "nadia", None])
    assert sorted(deduper._gen_similarity_pairs(array)) == [(0, 2), (1, 3)]

    assert list(lk.fuzzy(threshold=0.5)._gen_similarity_pairs(pa.array([None, None, None]))) == [(0, 1), (0, 2)]

    assert list(lk.fuzzy(threshold=0.5)._gen_similarity_pairs(pa.array([float("nan"), 1.0, None]))) == [(0, 2)]


def test_fuzzy_allpairs_two_values_only():
    values = ["london", "londn"]

    got = list(lk.fuzzy(threshold=0.8)._gen_similarity_pairs(pa.array(values)))

    scorer = lk.fuzzy().get_scorer()
    assert got == _brute_force_pairs(values, 0.8, scorer)


def test_fuzzy_allpairs_single_value_and_all_missing():
    assert list(lk.fuzzy(threshold=0.5)._gen_similarity_pairs(pa.array(["x"]))) == []
    assert list(lk.fuzzy(threshold=0.5)._gen_similarity_pairs(pa.array([None, None]))) == [(0, 1)]


############
# knobs    #
###########


def test_fuzzy_accepts_workers_and_max_block_memory():
    deduper = lk.fuzzy(workers=4, max_block_memory=1024)

    assert deduper._workers == 4
    assert deduper._max_block_memory == 1024


def test_fuzzy_knob_defaults_are_none():
    deduper = lk.fuzzy()

    assert deduper._workers is None
    assert deduper._max_block_memory is None


def test_fuzzy_rejects_bad_workers_value():
    with pytest.raises(InvalidArgumentValueError, match="`workers`"):
        lk.fuzzy(workers=0)


def test_fuzzy_rejects_non_integer_workers():
    with pytest.raises(InvalidArgumentTypeError, match="`workers` must be an integer, got"):
        lk.fuzzy(workers=2.0)


def test_fuzzy_rejects_bad_max_block_memory():
    with pytest.raises(InvalidArgumentValueError, match="`max_block_memory`"):
        lk.fuzzy(max_block_memory=-1)


def test_fuzzy_rejects_non_integer_max_block_memory():
    with pytest.raises(InvalidArgumentTypeError, match="`max_block_memory` must be an integer, got"):
        lk.fuzzy(max_block_memory=64.5)


def test_fuzzy_knobs_render_in_str():
    text = str(lk.fuzzy(workers=4, max_block_memory=2048))

    assert "workers=4" in text
    assert "max_block_memory=2048" in text


def test_fuzzy_explicit_workers_reaches_cdist(monkeypatch):
    """An explicit integer overrides the automatic resolution."""
    from rapidfuzz import process

    calls = []

    real_cdist = process.cdist

    def spy_cdist(queries, choices, **kwargs):
        if "workers" in kwargs:
            calls.append(kwargs["workers"])
        return real_cdist(queries, choices, **kwargs)

    monkeypatch.setattr(process, "cdist", spy_cdist)

    list(lk.fuzzy(threshold=0.8, workers=3)._gen_similarity_pairs(pa.array(["london", "londn", "paris"])))

    assert calls and all(call == 3 for call in calls)


def test_fuzzy_default_workers_local_is_cpu_count(monkeypatch):
    """`workers=None` resolves to all cores locally."""
    import os

    from rapidfuzz import process

    calls = []

    real_cdist = process.cdist

    def spy_cdist(queries, choices, **kwargs):
        if "workers" in kwargs:
            calls.append(kwargs["workers"])
        return real_cdist(queries, choices, **kwargs)

    monkeypatch.setattr(process, "cdist", spy_cdist)

    list(lk.fuzzy(threshold=0.8)._gen_similarity_pairs(pa.array(["london", "londn", "paris"])))

    assert calls and all(call == os.cpu_count() for call in calls)


def test_fuzzy_distributed_default_is_single_threaded(monkeypatch):
    """Under a distributed wrapper, `workers=None` resolves to 1."""
    from unittest.mock import Mock

    from rapidfuzz import process

    calls = []
    real_cdist = process.cdist

    def spy_cdist(queries, choices, **kwargs):
        if "workers" in kwargs:
            calls.append(kwargs["workers"])
        return real_cdist(queries, choices, **kwargs)

    monkeypatch.setattr(process, "cdist", spy_cdist)

    deduper = lk.fuzzy(threshold=0.8)
    deduper.set_frame(Mock())
    deduper.wdf.distributed = True

    list(deduper._gen_similarity_pairs(pa.array(["london", "londn", "paris"])))

    assert calls and all(call == 1 for call in calls)


def test_fuzzy_worker_resolution_not_used_on_topn_path(monkeypatch):
    """topn's path does not resolve workers: extract is single-threaded by design."""
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("liken.dedupers._blockwise.resolve_workers", fail)

    list(lk.fuzzy(threshold=0.7, topn=2)._gen_similarity_pairs(pa.array(["abcd", "abce", "abcf"])))

    assert calls == []
