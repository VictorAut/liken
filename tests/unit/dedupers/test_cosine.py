"""Liken deduper unit tests"""

import math

import numpy as np
import pyarrow as pa
import pytest

import liken as lk


def _normalized_matrix(table: pa.Table) -> np.ndarray:
    """Row-normalize a table the way `Cosine` does: zero-fill NaNs, guard zero norms."""
    columns = [table[col].to_numpy(zero_copy_only=False) for col in table.column_names]
    matrix = np.column_stack(columns)
    matrix = np.nan_to_num(matrix, nan=0.0)
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1
    return matrix / norms[:, None]


def _reference_pairs(table: pa.Table, threshold: float) -> list[tuple[int, int]]:
    """Verbatim copy of the pre-WI068 `Cosine._gen_similarity_pairs` loop."""
    normalized = _normalized_matrix(table)
    n = normalized.shape[0]
    pairs: list[tuple[int, int]] = []
    for i in range(n):
        sims = normalized[i] @ normalized[i + 1 :].T
        for offset, val in enumerate(sims):
            if val >= threshold:
                pairs.append((i, i + 1 + offset))
    return pairs


def _assert_matches_reference(table: pa.Table, threshold: float) -> None:
    expected = _reference_pairs(table, threshold)
    got = list(lk.cosine(threshold=threshold)._gen_similarity_pairs(table))
    assert got == expected


def _fixture_table(rng: np.random.Generator, n: int) -> pa.Table:
    """Random 3-column floats with duplicates, zero rows, and NaN-missing cells."""
    cols = []
    for _ in range(3):
        col = rng.random(n) * 100.0
        dupes = rng.integers(0, n, size=n // 10)
        col[dupes] = col[: n // 10]
        zeros = rng.random(n) < 0.05
        col[zeros] = 0.0
        missing = rng.random(n) < 0.05
        col[missing] = np.nan
        cols.append(col)
    return pa.table({"a": cols[0], "b": cols[1], "c": cols[2]})


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_cosine_matches_the_reference_loop_on_random_fixtures(seed: int):
    """Byte-identical output vs the replaced per-element loop."""
    rng = np.random.default_rng(seed)
    table = _fixture_table(rng, 120)
    _assert_matches_reference(table, 0.95)
    _assert_matches_reference(table, 0.5)
    _assert_matches_reference(table, 0.01)


def test_cosine_matches_the_reference_loop_with_all_zero_rows():
    """Zero-norm rows are guarded to unit vectors; they never match anything."""
    table = pa.table({"a": [0.0, 1.0, 0.0], "b": [0.0, 1.0, 0.0], "c": [0.0, 0.0, 0.0]})
    _assert_matches_reference(table, 0.95)


def test_cosine_matches_the_reference_loop_with_all_nan_rows():
    """All-NaN rows zero-fill to all-zero rows; the guard applies the same way."""
    table = pa.table({"a": [np.nan, 1.0, np.nan], "b": [np.nan, 1.0, np.nan], "c": [np.nan, 0.0, np.nan]})
    _assert_matches_reference(table, 0.95)


def test_cosine_matches_the_reference_loop_at_exact_threshold_duplicates():
    """Duplicate rows score exactly 1.0, an inclusive boundary match at a high threshold."""
    rng = np.random.default_rng(9)
    base = rng.random((40, 3))
    col_a = np.concatenate([base[:, 0], base[:, 0]])
    col_b = np.concatenate([base[:, 1], base[:, 1]])
    col_c = np.concatenate([base[:, 2], base[:, 2]])
    table = pa.table({"a": col_a, "b": col_b, "c": col_c})
    _assert_matches_reference(table, 0.95)


def test_cosine_matches_the_reference_loop_with_mixed_density():
    """Two near-identical clusters plus noise: both dense and sparse regions."""
    rng = np.random.default_rng(7)
    cluster = rng.random((1, 3))
    near = cluster + rng.random((15, 3)) * 1e-6
    far = rng.random((25, 3))
    rows = np.concatenate([near, far])
    table = pa.table({"a": rows[:, 0], "b": rows[:, 1], "c": rows[:, 2]})
    _assert_matches_reference(table, 0.95)


def test_cosine_yields_python_int_indices():
    """Tuple elements are Python ints, as the replaced enumerate loop emitted."""
    table = pa.table({"a": [1.0, 1.0], "b": [1.0, 1.0], "c": [1.0, 1.0]})

    pairs = list(lk.cosine(threshold=0.5)._gen_similarity_pairs(table))

    assert pairs == [(0, 1)]
    assert all(isinstance(x, int) for pair in pairs for x in pair)


def test_cosine_yields_i_ascending_then_j_ascending():
    """Yield order matches the replaced loop: i ascending, then j ascending."""
    rng = np.random.default_rng(5)
    base = rng.random((30, 3))
    rows = np.concatenate([base, base + rng.random((30, 3)) * 1e-9])
    table = pa.table({"a": rows[:, 0], "b": rows[:, 1], "c": rows[:, 2]})

    pairs = list(lk.cosine(threshold=0.95)._gen_similarity_pairs(table))

    i_indices = [i for i, _ in pairs]
    assert i_indices == sorted(i_indices)
    for i in set(i_indices):
        js = [j for pair_i, j in pairs if pair_i == i]
        assert js == sorted(js)


def test_cosine_matches_a_pair_exactly_at_the_threshold():
    """Rows [1, 1] and [1, 0] have similarity `1/sqrt(2)`; a threshold equal to it must match."""
    threshold = float(1.0 / np.sqrt(2.0))

    pairs = list(lk.cosine(threshold=threshold)._gen_similarity_pairs(pa.table({"x": [1.0, 1.0], "y": [1.0, 0.0]})))

    assert pairs == [(0, 1)]


def test_cosine_does_not_match_below_the_threshold():
    threshold = float(math.nextafter(1.0 / np.sqrt(2.0), math.inf))

    pairs = list(lk.cosine(threshold=threshold)._gen_similarity_pairs(pa.table({"x": [1.0, 1.0], "y": [1.0, 0.0]})))

    assert pairs == []
