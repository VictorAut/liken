"""Liken deduper unit tests"""

import math
from math import nextafter
from unittest.mock import Mock

import numpy as np
import pyarrow as pa
import pytest

import liken as lk
from liken.exceptions import InvalidArgumentTypeError
from liken.exceptions import InvalidArgumentValueError


################
# knn          #
################

# Mirrors the `test_fuzzy_topn_*` acceptance block (WI049): one
# mechanism, one policy — threshold required ∩ either-direction top-k.
# The tie test is the deliberate divergence: cKDTree's tie order is its
# own, not index order (WI049 D3 analog differs).


def test_knn_default_metric_is_euclidean():
    assert lk.knn(k=5)._metric == "euclidean"


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_knn_accepts_each_metric(metric):
    assert lk.knn(k=5, metric=metric)._metric == metric


def test_knn_unknown_metric_raises_value_error():
    with pytest.raises(InvalidArgumentValueError, match="`metric` must be one of"):
        lk.knn(k=5, metric="manhattan")


def test_knn_non_str_metric_raises_type_error():
    with pytest.raises(InvalidArgumentTypeError, match="`metric` must be of type 'str', got 'int'"):
        lk.knn(k=5, metric=1)


@pytest.mark.parametrize("k", [0, -1])
def test_knn_rejects_non_positive_k(k):
    """`k` below 1 is refused at construction: an unvalidated k reaches
    cKDTree and can abort the process."""
    with pytest.raises(InvalidArgumentValueError, match="`k` must be an integer in the range \\[1, inf\\), got"):
        lk.knn(k=k)


@pytest.mark.parametrize("k", [2.0, "2", True, None, []])
def test_knn_rejects_non_integer_k(k):
    """`k` must be a positive integer; there is no default.

    Unlike fuzzy's `topn`, `None` is rejected: `k` is required and has no
    absent state.
    """
    with pytest.raises(InvalidArgumentTypeError, match="`k` must be an integer, got"):
        lk.knn(k=k)


def test_knn_str_renders_threshold_k_and_metric():
    """`k` and `metric` always render: `k` has no absent state and
    `metric` has no default-free None."""
    assert "knn(threshold=0.95, k=5, metric='euclidean')" in str(lk.knn(k=5))
    assert "metric='cosine'" in str(lk.knn(k=5, metric="cosine"))


def test_knn_requires_compound_tuple_columns():
    with pytest.raises(InvalidArgumentTypeError):
        lk.knn(k=5).validate("name")

    with pytest.raises(InvalidArgumentTypeError):
        lk.knn(k=5).validate(("name", 3))


def test_knn_table_with_two_columns_pairs_close_rows():
    """Rows (0, 0) and (1, 0) sit at distance 1; rows 0 and 2 sit at
    distance 3. The euclidean similarity is 1/(1+d): 0.5 for the close
    pair, 0.25 for the far pair. At threshold 0.45 only the close pair
    matches."""
    table = pa.table({"x": [1.0, 2.0, 1.0], "y": [1.0, 1.0, 4.0]})

    pairs = list(lk.knn(threshold=0.45, k=1)._gen_similarity_pairs(table))

    assert pairs == [(0, 1)]


def test_knn_either_direction_emits_pair():
    """A pair is emitted when either side holds the other in its k best.

    Points on a line: A=0, B=1, C=10, k=1. A's best is B (d=1, s=0.5),
    B's best is A (d=1), C's best is B (d=9, s=0.1). At threshold 0.3,
    only (A,B) clears. At threshold 0.09, C's best partner B clears it
    (0.1) while B's best is still A: pair (B,C) survives through C's
    direction alone — the asymmetric case a one-direction-per-pair scan
    would miss. A does not hold C at any threshold here: C is not A's
    nearest.
    """
    table = pa.table({"x": [1.0, 2.0, 11.0]})

    pairs = sorted(lk.knn(threshold=0.3, k=1)._gen_similarity_pairs(table))

    assert pairs == [(0, 1)]

    pairs = sorted(lk.knn(threshold=0.09, k=1)._gen_similarity_pairs(table))

    assert pairs == [(0, 1), (1, 2)]


def test_knn_k_membership_is_per_row():
    """Each row keeps its own k best partners.

    Points: A=1, B=2, C=2.5, D=3.5, k=1. A's best is B (d=1, s=0.5), B's
    best is C (d=0.5, s ~ 0.667), C's best is B, D's best is C (d=1,
    s=0.5). At threshold 0.45, A-B, B-C and C-D all clear:
    chain emission. C-D survives through D's direction only — C's best
    is B, so a one-direction-per-pair scan from C would miss it. With
    k=2 the set is unchanged: A's second partner C sits at s=0.4, below
    the threshold. k is per row, and no row's k bounds another row's
    partner count.
    """
    table = pa.table({"x": [1.0, 2.0, 2.5, 3.5]})

    pairs = sorted(lk.knn(threshold=0.45, k=1)._gen_similarity_pairs(table))

    assert pairs == [(0, 1), (1, 2), (2, 3)]

    pairs = sorted(lk.knn(threshold=0.45, k=2)._gen_similarity_pairs(table))

    assert pairs == [(0, 1), (1, 2), (2, 3)]


@pytest.mark.parametrize(
    "threshold, expected",
    [(0.5, [(0, 1)]), (nextafter(0.5, 0.0), [(0, 1)]), (nextafter(0.5, 1.0), [])],
    ids=["exact-at-threshold", "just-below", "just-above"],
)
def test_knn_euclidean_boundary_in_the_similarity_domain(threshold, expected):
    """The euclidean comparison runs in the similarity domain: s = 1/(1+d).

    Two rows one unit apart give s = 0.5 exactly (the conversion is
    exact here): the threshold 0.5 matches, a threshold one ULP below
    also matches (0.5 >= it), a threshold one ULP above does not.
    Bare `>=` against the computed similarity, mirroring cosine: no
    tolerance band, the same inclusive-boundary contract as the compound
    numeric family.
    """
    table = pa.table({"x": [1.0, 2.0]})

    pairs = list(lk.knn(threshold=threshold, k=1)._gen_similarity_pairs(table))

    assert pairs == expected


def test_knn_cosine_boundary():
    """The cosine comparison runs on the pre-normalized conversion s = 1 - d^2/2.

    Rows [1, 0] and [0, 1] are orthogonal: after normalization d = sqrt(2),
    s = 1 - 2/2 = 0. The pair never matches on any positive threshold. Rows
    [1, 1] and [1, 0] have cosine similarity 1/sqrt(2): threshold equal to
    it matches. The documented numerical caveat bites at one ULP: the
    sqrt-then-square round trip through cKDTree leaves the converted
    similarity one ULP above the true cosine value, so the pair survives
    a one-ULP-higher threshold — one ULP looser than `lk.cosine`'s pinned
    boundary. Two ULPs above rejects, pinning that the drift is bounded
    at one ULP.
    """
    orthogonal = pa.table({"x": [1.0, 0.0], "y": [0.0, 1.0]})

    assert list(lk.knn(threshold=0.01, k=1, metric="cosine")._gen_similarity_pairs(orthogonal)) == []

    similar = pa.table({"x": [1.0, 1.0], "y": [1.0, 0.0]})
    exact = float(1.0 / np.sqrt(2.0))

    pairs = list(lk.knn(threshold=exact, k=1, metric="cosine")._gen_similarity_pairs(similar))

    assert pairs == [(0, 1)]

    one_up = float(math.nextafter(exact, math.inf))
    pairs = list(lk.knn(threshold=one_up, k=1, metric="cosine")._gen_similarity_pairs(similar))

    assert pairs == [(0, 1)]

    two_up = float(math.nextafter(one_up, math.inf))
    pairs = list(lk.knn(threshold=two_up, k=1, metric="cosine")._gen_similarity_pairs(similar))

    assert pairs == []


def test_knn_cosine_matches_all_pairs_cosine_at_large_k():
    """k >= n-1 degenerates to all-pairs; cosine knn matches lk.cosine exactly.

    The cross-check that the pre-normalized conversion reproduces cosine
    similarity on a non-boundary threshold (D2).
    """
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(12, 3))
    table = pa.table({f"col_{i}": matrix[:, i].tolist() for i in range(3)})

    threshold = 0.5

    knn_pairs = sorted(lk.knn(threshold=threshold, k=11, metric="cosine")._gen_similarity_pairs(table))
    cosine_pairs = sorted(lk.cosine(threshold=threshold)._gen_similarity_pairs(table))

    assert knn_pairs == cosine_pairs


def test_knn_ties_pinned_behaviour():
    """Ties at rank k follow cKDTree's own order, not index order.

    Five identical rows, k=2, threshold 0.99: every pair sits at d=0,
    s=1.0. cKDTree's tie order is deterministic per input but not
    index-ordered — every row's returned neighbours are rows 1 and 2
    (probed on scipy 1.16.3), so row 0 pairs with 1 and 2, row 1 with 2
    and 0, and rows 3 and 4 each reach only rows 1 and 2 within their
    k best. The pinned set is what scipy 1.16.3 returns; a scipy change
    that alters it fails this test visibly. Reordering the input can
    change the result: the same input order gives the same output,
    nothing more (WI017 concern). This is the deliberate divergence from
    the WI049 mirror: rapidfuzz breaks ties by index, cKDTree does not.
    """
    table = pa.table({"x": [1.0] * 5})

    pairs = sorted(lk.knn(threshold=0.99, k=2)._gen_similarity_pairs(table))

    assert pairs == [(0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4)]


def test_knn_exceeding_partners_matches_all_pairs():
    """k >= n - 1 degenerates to all-pairs.

    Pairwise similarities 1/(1+d): d(0,1)=1 -> 0.5; d(0,2)=3 -> 0.25;
    d(1,2)=sqrt(10) ~ 3.162 -> 0.2403. All three clear 0.2.
    """
    table = pa.table({"x": [1.0, 2.0, 1.0], "y": [1.0, 1.0, 4.0]})

    all_pairs = sorted(lk.knn(threshold=0.2, k=10)._gen_similarity_pairs(table))

    assert all_pairs == [(0, 1), (0, 2), (1, 2)]


def test_knn_zero_norm_rows_match_nothing():
    """An all-zero row matches nothing, under both metrics.

    The spec pins "a row whose norm is 0 is normalised as a zero vector,
    so it matches nothing"; the clamp alone would put two zero rows at
    d=0, s=1.0 under both conversions, so zero-norm rows are kept out of
    the tree entirely: they neither emit nor appear in another row's k
    best. Here rows 0 and 1 are all-zero and excluded; row 2 never pairs
    with them even though euclidean d(row 2, row 0) = sqrt(2) would
    clear the threshold, and under cosine the clamped row would sit at
    similarity 0.5 from everything.
    """
    all_zero = pa.table({"x": [0.0, 0.0, 1.0], "y": [0.0, 0.0, 1.0]})

    for metric in ("euclidean", "cosine"):
        pairs = list(lk.knn(threshold=0.01, k=2, metric=metric)._gen_similarity_pairs(all_zero))

        assert pairs == []
        assert all(0 not in pair and 1 not in pair for pair in pairs)


def test_knn_infinity_rows_match_nothing():
    """A row carrying an infinity matches nothing, under both metrics.

    The 0.0 fill leaves infinities in place. An inf row's norm is inf
    (not 0, so a zero-norm mask misses it); under cosine it normalizes
    to an exact zero vector, which sits at d=1, s=0.5 from every unit
    vector and would silently match anything below a 0.5 threshold —
    diverging from `lk.cosine`, whose 0.0 fill makes inf rows match
    nothing. Non-finite rows are excluded from the tree like zero-norm
    rows.
    """
    with_inf = pa.table({"x": [1.0, 1.0, float("inf")], "y": [0.0, 1.0, 1.0]})

    for metric in ("euclidean", "cosine"):
        pairs = list(lk.knn(threshold=0.4, k=2, metric=metric)._gen_similarity_pairs(with_inf))

        assert pairs == [(0, 1)]  # the value pair; the inf row joins nothing


def test_knn_nulls_filled_like_cosine():
    """A missing member is filled with 0.0, mirroring cosine.

    Rows [1, 1] and [None, 1] compare as [1, 1] vs [0, 1] after the
    fill: euclidean distance 1, similarity 1/(1+1) = 0.5, matched at 0.4
    and not at 0.51. Cosine on the same fill: rows normalize to
    [1/sqrt(2), 1/sqrt(2)] and [0, 1]; cos = 1/sqrt(2) ~ 0.707, matched
    at 0.4 and not at 0.8.
    """
    table = pa.table({"x": [1.0, None], "y": [1.0, 1.0]})

    pairs = list(lk.knn(threshold=0.4, k=1)._gen_similarity_pairs(table))

    assert pairs == [(0, 1)]

    pairs = list(lk.knn(threshold=0.51, k=1)._gen_similarity_pairs(table))

    assert pairs == []

    pairs = list(lk.knn(threshold=0.4, k=1, metric="cosine")._gen_similarity_pairs(table))

    assert pairs == [(0, 1)]

    pairs = list(lk.knn(threshold=0.8, k=1, metric="cosine")._gen_similarity_pairs(table))

    assert pairs == []


def test_knn_query_called_once_per_frame(monkeypatch):
    """The mechanism is one vectorized query, not a per-pair scan.

    `cKDTree.query` runs exactly once for the whole frame, with
    `k = self._k + 1`. This is the no-all-pairs-scan structural test,
    mirroring WI049's extract spy. `cKDTree` is immutable, so the spy
    patches the module-level name `liken.dedupers.knn` binds, the same
    seam as the edit_distance cdist spy.
    """
    import liken.dedupers.knn as knn_module

    real_tree = knn_module.cKDTree
    calls = []

    class SpyTree(real_tree):
        def query(self, x, k=1, **kwargs):
            calls.append({"k": k})
            return super().query(x, k=k, **kwargs)

    monkeypatch.setattr(knn_module, "cKDTree", SpyTree)

    table = pa.table({"x": [1.0, 2.0, 3.0, None]})

    pairs = sorted(lk.knn(threshold=0.1, k=1)._gen_similarity_pairs(table))

    assert pairs == [(0, 1), (1, 2)]
    assert len(calls) == 1
    assert calls[0]["k"] == 2  # k + 1


def test_knn_single_column_table():
    """One numeric column through the compound machinery works (1-D).

    The research's noted overlap with `within`: coexistence, not
    deduplication.
    """
    table = pa.table({"x": [1.0, 2.0, 3.0]})

    pairs = sorted(lk.knn(threshold=0.3, k=1)._gen_similarity_pairs(table))

    assert pairs == [(0, 1), (1, 2)]


def test_knn_transitivity_through_union_find(mock_df):
    """Merged pairs chain transitively through union-find.

    Points A=0, B=1, C=2, k=1, threshold 0.4: pairs (A,B) and (B,C)
    qualify at 1/(1+1) = 0.5, A-C never does (d=2, s ~ 0.33). Union-find
    still merges all three through B — the docstring's transitivity note.
    """
    mock_df.get_array = Mock(return_value=pa.table({"x": [1.0, 2.0, 3.0]}))

    uf, n = lk.knn(threshold=0.4, k=1).set_frame(mock_df).build_union_find(("x",), [])

    assert n == 3
    assert uf[0] == uf[1] == uf[2]
