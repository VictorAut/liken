"""Unit tests for `jaccard`'s Gram-strip pair path, knobs and null contract."""

import math
import random

import pyarrow as pa
import pytest

import liken as lk
from liken.exceptions import InvalidArgumentTypeError
from liken.exceptions import InvalidArgumentValueError


def _table_from_rows(rows, cols):
    """Build a compound-column table: string columns plus one NaN carrier.

    pyarrow infers per-column types from mixed cells and rejects a column
    mixing strings and floats, so None lives in the string columns and a
    separate float column carries the NaN members.

    Returns:
        The table and the full rows including the carrier column's values,
        so reference comparisons see exactly what the deduper sees.
    """
    rng = random.Random(99)
    nan_values = [float("nan") if rng.random() < 0.15 else 1.0 for _ in rows]
    data = {f"c{i}": pa.array([row[i] for row in rows], type=pa.string()) for i in range(cols)}
    data["nan_col"] = pa.array(nan_values, type=pa.float64())
    return pa.table(data), [[*row, value] for row, value in zip(rows, nan_values, strict=True)]


def _reference_pairs(rows, threshold):
    """Pure-Python reference under option C: None and NaN are missing."""
    sets = [{v for v in row if v is not None and v == v} for row in rows]
    out = []
    n = len(sets)
    for i in range(n):
        for j in range(i + 1, n):
            intersection = sets[i] & sets[j]
            if not intersection:
                continue
            if len(intersection) / len(sets[i] | sets[j]) >= threshold:
                out.append((i, j))
    return out


def _seeded_rows(n, *, seed=21, vocab=10, cols=3):
    """Deterministic compound rows with None cells.

    Cells are strings with None as the missing value; the table helper
    adds a separate float column carrying the NaN members.
    """
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        rows.append([f"v{rng.randrange(vocab)}" if rng.random() >= 0.15 else None for _ in range(cols)])
    return rows


def test_jaccard_byte_identical_to_reference():
    rows = _seeded_rows(40)
    table, full_rows = _table_from_rows(rows, 3)

    expected = _reference_pairs(full_rows, 0.5)
    got = list(lk.jaccard(threshold=0.5)._gen_similarity_pairs(table))

    assert got == expected


def test_jaccard_small_cap_matches_the_reference():
    rows = _seeded_rows(30, seed=5)
    table, full_rows = _table_from_rows(rows, 3)

    expected = _reference_pairs(full_rows, 0.5)
    got = list(lk.jaccard(threshold=0.5, max_block_memory=1)._gen_similarity_pairs(table))

    assert got == expected


def test_jaccard_mixed_none_and_nan_within_a_row_are_dropped():
    """Option C: both None and NaN are missing, neither matches nor counts.

    Row 0 keeps only {a}: None and NaN contribute nothing, so it scores
    jaccard 0.5 against row 1's {a, b}. Under the pre-WI063 behaviour the
    NaN and None members stayed in the set and the same pair scored 0.25,
    so threshold 0.4 distinguishes the contract.
    """
    table = pa.table({"x": ["a", "a"], "y": [None, "b"], "z": pa.array([float("nan"), None], type=pa.float64())})

    pairs = list(lk.jaccard(threshold=0.4)._gen_similarity_pairs(table))

    assert pairs == [(0, 1)]


def test_jaccard_all_missing_row_matches_nothing():
    """A row whose members are all missing has no members, so no pairs."""
    table = pa.table(
        {
            "x": pa.array([None, "a", None], type=pa.string()),
            "y": pa.array([float("nan"), 1.0, float("nan")], type=pa.float64()),
        }
    )

    assert list(lk.jaccard(threshold=0.01)._gen_similarity_pairs(table)) == []


def test_jaccard_nan_never_matches_a_value():
    """A NaN member is dropped, not a token: it matches nothing, anywhere.

    Two rows holding only NaN members have no members at all, so no pair
    matches; a NaN never counts as a shared member.
    """
    table = pa.table({"x": pa.array([float("nan"), float("nan")], type=pa.float64())})

    assert list(lk.jaccard(threshold=0.01)._gen_similarity_pairs(table)) == []


def test_jaccard_two_all_null_rows_do_not_match():
    table = pa.table({"x": [None, None], "y": [None, None]})

    assert list(lk.jaccard(threshold=0.01)._gen_similarity_pairs(table)) == []


########
# knobs #
#######


def test_jaccard_accepts_max_block_memory():
    deduper = lk.jaccard(max_block_memory=4096)

    assert deduper._max_block_memory == 4096


def test_jaccard_max_block_memory_defaults_to_none():
    assert lk.jaccard()._max_block_memory is None


def test_jaccard_rejects_negative_max_block_memory():
    with pytest.raises(InvalidArgumentValueError, match="`max_block_memory`"):
        lk.jaccard(max_block_memory=-1)


def test_jaccard_accepts_zero_max_block_memory():
    """A zero cap clamps to one-row blocks at generation time, never raises."""
    deduper = lk.jaccard(max_block_memory=0)
    assert deduper._max_block_memory == 0
    table = pa.table({"x": ["a", "a"], "y": ["b", "b"]})
    assert list(deduper._gen_similarity_pairs(table)) == [(0, 1)]


def test_jaccard_rejects_non_integer_max_block_memory():
    with pytest.raises(InvalidArgumentTypeError, match="`max_block_memory` must be an integer, got"):
        lk.jaccard(max_block_memory=64.5)


def test_jaccard_max_block_memory_renders_in_str():
    assert "max_block_memory=4096" in str(lk.jaccard(max_block_memory=4096))


def test_jaccard_boundary_pair_exactly_at_the_threshold():
    """A pair scoring exactly 0.5 still matches through the Gram path."""
    table = pa.table({"x": ["a", "a"], "y": ["b", "b"], "z": ["c", "d"]})

    assert list(lk.jaccard(threshold=0.5)._gen_similarity_pairs(table)) == [(0, 1)]
    below = math.nextafter(0.5, math.inf)
    assert list(lk.jaccard(threshold=below)._gen_similarity_pairs(table)) == []
