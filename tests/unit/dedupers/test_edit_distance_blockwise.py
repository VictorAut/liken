"""Unit tests for `edit_distance`'s block-wise pair path and its knobs."""

import random

import pyarrow as pa
import pytest
from rapidfuzz.distance import Levenshtein

import liken as lk
from liken.dedupers.edit_distance import _BOUNDARY_TOLERANCE
from liken.exceptions import InvalidArgumentTypeError
from liken.exceptions import InvalidArgumentValueError


def _brute_force_pairs(values, max_distance, metric):
    """Pure-Python reference: the per-row loop's pair output, in yield order.

    Mirrors edit_distance.py's pre-change loop, including the missing-value
    star pairs and the float64 branch for the normalized metrics.
    """
    from rapidfuzz import process
    from rapidfuzz.distance import DamerauLevenshtein
    from rapidfuzz.distance import Hamming
    from rapidfuzz.distance import Indel
    from rapidfuzz.distance import JaroWinkler

    from liken.core.missing import partition_missing

    # metric -> (scorer, runs on dtype="float64"), mirroring edit_distance.py
    metrics = {
        "levenshtein": (Levenshtein.distance, False),
        "damerau_levenshtein": (DamerauLevenshtein.normalized_distance, True),
        "jaro_winkler": (JaroWinkler.normalized_distance, True),
        "indel": (Indel.normalized_distance, True),
        "hamming": (Hamming.normalized_distance, True),
    }

    scorer, float64 = metrics[metric]

    missing, present = partition_missing(values)
    pairs = [(missing[0], i) for i in missing[1:]]

    scored = [values[i] for i in present]

    for pos, s1 in enumerate(scored):
        if pos + 1 >= len(scored):
            break
        if float64:
            distances = process.cdist([s1], scored[pos + 1 :], scorer=scorer, dtype="float64")[0]
            bound = max_distance + _BOUNDARY_TOLERANCE
        else:
            distances = process.cdist([s1], scored[pos + 1 :], scorer=scorer)[0]
            bound = max_distance
        for offset, distance in enumerate(distances):
            if distance > bound:
                continue
            pairs.append((present[pos], present[pos + 1 + offset]))

    return pairs


def _seeded_values(n, seed=4321):
    """Deterministic short-code values with near-duplicates and nulls."""
    rng = random.Random(seed)
    base = ["OL5 9PL", "M1 2WD", "EH1 1AA", "SW1A 0AA", "B33 8TH"]
    values = []
    for _ in range(n):
        if values and rng.random() < 0.4:
            values.append(base[rng.randrange(len(base))])
        else:
            values.append(f"{rng.randrange(10)}{rng.randrange(10)}{rng.randrange(10)} {rng.randrange(10)}AB")
    for i in range(n):
        if rng.random() < 0.1:
            values[i] = None
    return values


#############################
# byte-identical pair output #
############################


@pytest.mark.parametrize("metric", ["levenshtein", "damerau_levenshtein", "jaro_winkler", "indel", "hamming"])
def test_edit_distance_byte_identical_to_reference(metric):
    """Both branches (integer levenshtein, normalized float64) reproduce the
    per-row loop's output exactly."""
    values = _seeded_values(50)
    max_distance = 2 if metric == "levenshtein" else 0.15

    expected = _brute_force_pairs(values, max_distance, metric)
    got = list(lk.edit_distance(max_distance=max_distance, metric=metric)._gen_similarity_pairs(pa.array(values)))

    assert got == expected


def test_edit_distance_blockwise_cap_forces_multiple_blocks():
    """A tiny cap still yields identical output, across several cdist calls."""
    values = _seeded_values(40)

    expected = _brute_force_pairs(values, 2, "levenshtein")
    got = list(lk.edit_distance(max_block_memory=1)._gen_similarity_pairs(pa.array(values)))

    assert got == expected


def test_edit_distance_boundary_distance_matches():
    """A distance exactly at the inclusive bound matches through the block path."""
    values = ["abc", "abcx", "abcxy"]

    pairs = list(lk.edit_distance(max_distance=1)._gen_similarity_pairs(pa.array(values)))

    assert pairs == [(0, 1), (1, 2)]


def test_edit_distance_boundary_tolerance_survives_blocks():
    """The normalized metrics' tolerance survives the block path: a distance
    within `1e-9` above the bound still matches, as in the loop."""
    from rapidfuzz import process
    from rapidfuzz.distance import Indel

    values = ["abcd", "abce"]
    matrix = process.cdist(["abcd"], ["abce"], scorer=Indel.normalized_distance, dtype="float64")
    distance = float(matrix[0][0])
    # a bound sitting `tolerance/2` below the actual distance admits it
    # only through the tolerance
    max_distance = distance - _BOUNDARY_TOLERANCE / 2

    deduper = lk.edit_distance(max_distance=max_distance, metric="indel")
    got = list(deduper._gen_similarity_pairs(pa.array(values)))

    assert got == [(0, 1)]


def test_edit_distance_nulls_and_missing_shapes():
    """Null star pairs, null-value separation and NaN handling are unchanged."""
    deduper = lk.edit_distance(max_distance=2)

    array = pa.array(["nathan", None, "nadia", None])
    assert sorted(deduper._gen_similarity_pairs(array)) == [(1, 3)]

    assert list(lk.edit_distance()._gen_similarity_pairs(pa.array([None, None, None]))) == [(0, 1), (0, 2)]

    assert list(lk.edit_distance()._gen_similarity_pairs(pa.array([float("nan"), 1.0, None]))) == [(0, 2)]


def test_edit_distance_single_value_and_all_missing():
    assert list(lk.edit_distance()._gen_similarity_pairs(pa.array(["x"]))) == []
    assert list(lk.edit_distance()._gen_similarity_pairs(pa.array([None, None]))) == [(0, 1)]


########
# knobs #
#######


def test_edit_distance_accepts_workers_and_max_block_memory():
    deduper = lk.edit_distance(workers=4, max_block_memory=1024)

    assert deduper._workers == 4
    assert deduper._max_block_memory == 1024


def test_edit_distance_knob_defaults_are_none():
    deduper = lk.edit_distance()

    assert deduper._workers is None
    assert deduper._max_block_memory is None


def test_edit_distance_rejects_bad_workers_value():
    with pytest.raises(InvalidArgumentValueError, match="`workers`"):
        lk.edit_distance(workers=0)


def test_edit_distance_rejects_non_integer_workers():
    with pytest.raises(InvalidArgumentTypeError, match="`workers` must be an integer, got"):
        lk.edit_distance(workers=2.0)


def test_edit_distance_rejects_negative_max_block_memory():
    with pytest.raises(InvalidArgumentValueError, match="`max_block_memory`"):
        lk.edit_distance(max_block_memory=-1)


def test_edit_distance_rejects_non_integer_max_block_memory():
    with pytest.raises(InvalidArgumentTypeError, match="`max_block_memory` must be an integer, got"):
        lk.edit_distance(max_block_memory=64.5)


def test_edit_distance_knobs_render_in_str():
    text = str(lk.edit_distance(workers=2, max_block_memory=4096))

    assert "workers=2" in text
    assert "max_block_memory=4096" in text


def test_edit_distance_explicit_workers_reaches_cdist(monkeypatch):
    """An explicit integer overrides the automatic resolution."""
    from rapidfuzz import process

    calls = []
    real_cdist = process.cdist

    def spy_cdist(queries, choices, **kwargs):
        if "workers" in kwargs:
            calls.append(kwargs["workers"])
        return real_cdist(queries, choices, **kwargs)

    monkeypatch.setattr(process, "cdist", spy_cdist)

    list(lk.edit_distance(workers=5)._gen_similarity_pairs(pa.array(["abc", "abd", "abe"])))

    assert calls and all(call == 5 for call in calls)


def test_edit_distance_distributed_default_is_single_threaded(monkeypatch):
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

    deduper = lk.edit_distance()
    deduper.set_frame(Mock())
    deduper.wdf.distributed = True

    list(deduper._gen_similarity_pairs(pa.array(["abc", "abd", "abe"])))

    assert calls and all(call == 1 for call in calls)
