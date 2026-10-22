"""Liken deduper unit tests"""

import math
from decimal import Decimal
from unittest.mock import Mock

import pyarrow as pa
import pytest

import liken as lk
from liken.exceptions import InvalidArgumentTypeError
from liken.exceptions import InvalidArgumentValueError


##################################
# sorted_neighbourhood          #
##################################

# WI051: threshold deduper over a sorted-order window. The pair set is a
# strict subset of all-pairs `fuzzy` at the same threshold and scorer:
# same acceptance semantics, pair generation restricted to the window.


def _stub_score_table(table):
    """A scorer stub driven by a value-keyed score table.

    Accepts `score_cutoff` as a kwarg: `process.extract` passes it. The
    stub returns the table score for the pair, or 0.0 for unlisted pairs.
    """

    def scorer(s1, s2, score_cutoff=None):
        return table.get((s1, s2), table.get((s2, s1), 0.0))

    return scorer


def test_sorted_neighbourhood_window_limits_pairs():
    """`window=w` scores each row only against the next w-1 rows in sort order.

    Sorted layout: "abcd"(0) < "abcdz"(3) < "abce"(1) < "abcf"(2), input
    order [abcd, abce, abcf, abcdz]. fuzz.ratio scores: abcd-abcdz 88.9,
    abcdz-abce 66.7, abcdz-abcf 66.7, abcd-abce 75, abce-abcf 75,
    abcd-abcf 75. At threshold 0.75, window=2 scores only consecutive
    sorted pairs — abcd-abcdz (88.9) and abce-abcf (75) — so the pairs
    are (0,3) and (1,2). window=3 also scores abcd-abce (75): (0,1)
    joins. The abcdz-abce/abcf pairs (66.7) stay unscored even at
    window=3; all-pairs fuzzy at the same threshold emits them all.
    """
    array = pa.array(["abcd", "abce", "abcf", "abcdz"])

    pairs_w2 = sorted(lk.sorted_neighbourhood(threshold=0.75, window=2)._gen_similarity_pairs(array))
    pairs_w3 = sorted(lk.sorted_neighbourhood(threshold=0.75, window=3)._gen_similarity_pairs(array))

    assert pairs_w2 == [(0, 3), (1, 2)]
    assert pairs_w3 == [(0, 1), (0, 3), (1, 2)]

    all_pairs = sorted(lk.fuzzy(threshold=0.75)._gen_similarity_pairs(array))
    assert all_pairs == [(0, 1), (0, 2), (0, 3), (1, 2)]


def test_sorted_neighbourhood_scores_in_sorted_order():
    """Similar values pair when they are adjacent in sorted order, not input order.

    Input order [b, a, c] with a,b,c similar to each other in sorted
    adjacency: sorted order is aaa, aab, aac; consecutive pairs
    aaa-aab and aab-aac are scored. The pair (aaa, aac) sits two apart
    in sort order, so window=2 cannot emit it even though input order
    would have put aac adjacent to nothing relevant.
    """
    array = pa.array(["aab", "aaa", "aac"])

    pairs = sorted(lk.sorted_neighbourhood(threshold=0.6, window=2)._gen_similarity_pairs(array))

    # sorted: aaa(1), aab(0), aac(2); adjacent pairs (1,0)->(0,1) and (0,2)
    assert pairs == [(0, 1), (0, 2)]


def test_sorted_neighbourhood_pairs_are_subset_of_fuzzy():
    """Every SN pair is an all-pairs fuzzy pair at the same threshold/scorer.

    With window >= n the sets are equal: the window no longer restricts
    anything.
    """
    array = pa.array(["london", "londn", "lyon", "lyonn", None])

    for threshold in (0.6, 0.8):
        fuzzy_pairs = set(lk.fuzzy(threshold=threshold)._gen_similarity_pairs(array))
        sn_pairs = set(lk.sorted_neighbourhood(threshold=threshold, window=len(array))._gen_similarity_pairs(array))
        assert sn_pairs <= fuzzy_pairs
        assert sn_pairs == fuzzy_pairs


def test_sorted_neighbourhood_pair_direction_matches_fuzzy(monkeypatch):
    """Each in-window pair is scored in fuzzy's original-index direction.

    fuzzy scores the pair (earlier original index -> later original
    index). The stub scorer below is asymmetric by construction — the
    real scorers in the registry happen to be symmetric (probed:
    partial_ratio("abcd", "xxx abcd yyy") = 100 both ways on
    rapidfuzz), so a stub is what makes the direction observable. A
    sorted-direction implementation would score s(B, A) here and emit
    the pair; fuzzy's direction scores s(A, B) = 80 and rejects s(B, A)
    = 0.
    """
    directed = {("A", "B"): 80.0}

    def scorer(s1, s2, score_cutoff=None):
        return directed.get((s1, s2), 0.0)

    deduper = lk.sorted_neighbourhood(threshold=0.75, window=2)
    monkeypatch.setattr(deduper, "get_scorer", lambda: scorer)

    # input [A, B]: sorted order is A, B (A < B); the pair is scored
    # A->B (fuzzy's direction) = 80 >= 75: emitted
    pairs = sorted(deduper._gen_similarity_pairs(pa.array(["A", "B"])))
    assert pairs == [(0, 1)]

    # input [B, A]: same sorted layout, but fuzzy's direction is B->A
    # = 0 < 75: NOT emitted
    pairs = sorted(deduper._gen_similarity_pairs(pa.array(["B", "A"])))
    assert pairs == []

    # and all-pairs fuzzy agrees on both orders
    fuzzy = lk.fuzzy(threshold=0.75)
    monkeypatch.setattr(fuzzy, "get_scorer", lambda: scorer)
    assert sorted(fuzzy._gen_similarity_pairs(pa.array(["A", "B"]))) == [(0, 1)]
    assert sorted(fuzzy._gen_similarity_pairs(pa.array(["B", "A"]))) == []


def test_sorted_neighbourhood_boundary(monkeypatch):
    """Exact-at-threshold and within-tolerance emission on the 0-100 scale.

    A stubbed `process.cdist` returning `100 * threshold + drift` pins
    the guard itself (the edit_distance drift tests' convention): at the
    exact boundary and within the 1e-9 tolerance the pair is emitted,
    beyond it is not. A stub is needed because real `cdist` rounds to
    float32 by default, which absorbs the 2e-9 drift. The real-string
    case pins inclusivity: fuzz.ratio("abcd", "abce") is exactly 75, so
    threshold 0.75 matches and 0.76 does not.
    """
    from rapidfuzz import process as fuzz_process

    array = pa.array(["abcd", "abce"])

    assert list(lk.sorted_neighbourhood(threshold=0.75, window=2)._gen_similarity_pairs(array)) == [(0, 1)]
    assert list(lk.sorted_neighbourhood(threshold=0.76, window=2)._gen_similarity_pairs(array)) == []

    threshold = 0.75
    for drift, expected in ((0.0, [(0, 1)]), (-5e-10, [(0, 1)]), (-2e-9, [])):
        score = float(Decimal(str(threshold)) * 100) + drift
        deduper = lk.sorted_neighbourhood(threshold=threshold, window=2)
        monkeypatch.setattr(fuzz_process, "cdist", lambda queries, choices, score=score, **kwargs: [[score]])
        assert list(deduper._gen_similarity_pairs(pa.array(["x", "y"]))) == expected


def test_sorted_neighbourhood_scorer_registry_is_fuzzys():
    """The scorer registry is fuzzy's: every name resolves to the same callable."""
    from liken.dedupers.fuzzy import _SCORER_NAMES
    from liken.dedupers.fuzzy import _SCORERS

    for name in _SCORER_NAMES:
        deduper = lk.sorted_neighbourhood(scorer=name, window=2)
        assert deduper.get_scorer() is _SCORERS[name]
        assert deduper.get_scorer() is lk.fuzzy(scorer=name).get_scorer()


def test_sorted_neighbourhood_unknown_scorer_raises():
    with pytest.raises(InvalidArgumentValueError, match="`scorer` must be one of"):
        lk.sorted_neighbourhood(scorer="nope", window=2)


def test_sorted_neighbourhood_jaro_winkler_matches_at_scaled_threshold():
    """The jaro_winkler wrapper works under the windowed path.

    Same fixture and thresholds as the fuzzy jaro_winkler pins.
    """
    s1 = "Calle Ancho, 12, 05688, Rioja, Navarra, Espana"
    s2 = "Calle Sueco, 56, 05688, Rioja, Navarra"

    # sorted order: "Calle Ancho..." < "Calle Sueco...", adjacent
    pairs = list(
        lk.sorted_neighbourhood(threshold=0.9, window=2, scorer="jaro_winkler")._gen_similarity_pairs(
            pa.array([s1, s2])
        )
    )

    assert pairs == [(0, 1)]

    pairs = list(
        lk.sorted_neighbourhood(threshold=0.91, window=2, scorer="jaro_winkler")._gen_similarity_pairs(
            pa.array([s1, s2])
        )
    )

    assert pairs == []


# missing values, mirroring the fuzzy null block


def test_sorted_neighbourhood_nulls_pair_with_nulls_only():
    """Missing values pair with each other, never with a value.

    Sorted layout: "nadia"(2) < "nathan"(0), adjacent, and
    fuzz.ratio("nathan", "nadia") is ~54.5 >= 50, so the value pair also
    fires at window=2. The null star pair (1, 3) forms regardless.
    """
    array = pa.array(["nathan", None, "nadia", None])

    pairs = sorted(lk.sorted_neighbourhood(threshold=0.5, window=2)._gen_similarity_pairs(array))

    assert pairs == [(0, 2), (1, 3)]


def test_sorted_neighbourhood_null_never_pairs_with_value():
    """At any threshold a null pairs only with another null.

    Near-zero threshold: every in-window value pair also fires (the
    fuzzy near-zero pin emitted (0, 2) as well); the assertion is that
    no pair crosses the null/value divide.
    """
    array = pa.array(["nathan", None, "nadia", None])

    pairs = sorted(lk.sorted_neighbourhood(threshold=math.nextafter(0.0, 1.0), window=4)._gen_similarity_pairs(array))

    assert (1, 3) in pairs
    assert all(1 not in pair and 3 not in pair for pair in pairs if pair != (1, 3))


def test_sorted_neighbourhood_all_null_input_yields_star_pairs():
    array = pa.array([None, None, None])

    assert list(lk.sorted_neighbourhood(threshold=0.5, window=2)._gen_similarity_pairs(array)) == [(0, 1), (0, 2)]


def test_sorted_neighbourhood_nan_and_null_pair():
    """NaN is missing and pairs with a null, never with a value."""
    array = pa.array([float("nan"), 1.0, None])

    assert list(lk.sorted_neighbourhood(threshold=0.5, window=2)._gen_similarity_pairs(array)) == [(0, 2)]


def test_sorted_neighbourhood_cdist_receives_no_missing_values(monkeypatch):
    """process.cdist scores non-missing values only."""
    from rapidfuzz import process

    from liken.core.missing import is_missing

    real_cdist = process.cdist

    def spy_cdist(queries, choices, **kwargs):
        assert all(not is_missing(value) for value in (*queries, *choices))
        return real_cdist(queries, choices, **kwargs)

    monkeypatch.setattr(process, "cdist", spy_cdist)

    array = pa.array(["nathan", None, "nadia", None])

    assert list(lk.sorted_neighbourhood(threshold=0.95, window=2)._gen_similarity_pairs(array)) == [(1, 3)]


def test_sorted_neighbourhood_scored_pairs_bounded_by_window(monkeypatch):
    """Scored pair count is bounded by n*(window-1): no all-pairs scan.

    `process.cdist` calls carry the per-call query count and choice
    count; the sum of query*choice over all calls must stay within the
    window bound, and no single call may score more than window-1
    choices.
    """
    from rapidfuzz import process

    real_cdist = process.cdist
    calls = []

    def spy_cdist(queries, choices, **kwargs):
        calls.append((len(queries), len(choices)))
        return real_cdist(queries, choices, **kwargs)

    monkeypatch.setattr(process, "cdist", spy_cdist)

    values = ["val-01", "val-02", "val-03", "val-04", "val-05", "val-06"]
    array = pa.array(values)

    window = 3
    pairs = list(lk.sorted_neighbourhood(threshold=0.1, window=window)._gen_similarity_pairs(array))

    scored = sum(q * c for q, c in calls)
    n_present = len(values)
    assert scored <= n_present * (window - 1)
    assert all(c <= window - 1 for _q, c in calls)
    # window=3 covers every sorted-adjacent and skip-1 pair; all 15
    # distinct pairs would need window=6
    assert len(pairs) < len(values) * (len(values) - 1) // 2


def test_sorted_neighbourhood_tie_group_window_edge():
    """Equal sort keys cut differently at window edges: deterministic per input order.

    Identical strings score 100 against each other, but window=2 only
    scores consecutive sorted positions, so inside a tie group larger
    than the window some identical pairs are never scored. Which index
    pairs survive depends on input order (stable sort keeps ties in
    input order). Deterministic per input order, NOT
    permutation-invariant (WI017's concern) — unlike all-pairs fuzzy,
    which emits every identical pair.
    """
    # Input order A: sorted order is aa(0), aa(2), aa(4), ab(1), ac(3)
    # — the scored pairs are (0,2), (2,4), (1,3), (1,4); two of them
    # are "aa"-"aa". What window=2 never scores is the outer "aa"-"aa"
    # pair (0, 4), which all-pairs fuzzy emits.
    array_a = pa.array(["aa", "ab", "aa", "ac", "aa"])
    pairs_a = sorted(lk.sorted_neighbourhood(threshold=0.5, window=2)._gen_similarity_pairs(array_a))

    # Input order B: the three "aa" occupy sorted positions 0, 1, 2 —
    # every consecutive pair among them is scored.
    array_b = pa.array(["aa", "aa", "aa", "ab", "ac"])
    pairs_b = sorted(lk.sorted_neighbourhood(threshold=0.5, window=2)._gen_similarity_pairs(array_b))

    assert pairs_a != pairs_b

    # A re-run on the same input gives the same answer (deterministic).
    pairs_a2 = sorted(lk.sorted_neighbourhood(threshold=0.5, window=2)._gen_similarity_pairs(array_a))
    assert pairs_a2 == pairs_a

    # At a window covering the whole column the sets agree with fuzzy.
    sn_full = set(lk.sorted_neighbourhood(threshold=0.5, window=5)._gen_similarity_pairs(array_a))
    fuzzy_pairs = set(lk.fuzzy(threshold=0.5)._gen_similarity_pairs(array_a))
    assert sn_full == fuzzy_pairs


def test_sorted_neighbourhood_transitivity_through_union_find(mock_df, monkeypatch):
    """Merged pairs chain transitively through union-find.

    Values sorted: A < B < C, consecutive similarities above threshold,
    A-C below. window=2 scores only the adjacent pairs; union-find still
    merges all three through B.
    """
    scores = {("A", "B"): 80.0, ("A", "C"): 60.0, ("B", "C"): 75.0}

    deduper = lk.sorted_neighbourhood(threshold=0.7, window=2)
    monkeypatch.setattr(deduper, "get_scorer", lambda: _stub_score_table(scores))

    mock_df.get_array = Mock(return_value=pa.array(["B", "A", "C"]))

    uf, n = deduper.set_frame(mock_df).build_union_find("address", [])

    assert n == 3
    assert uf[0] == uf[1] == uf[2]


def test_sorted_neighbourhood_rejects_non_positive_window():
    """`window` below 2 scores nothing: refused at construction."""
    with pytest.raises(InvalidArgumentValueError, match="`window` must be an integer in the range \\[2, inf\\), got"):
        lk.sorted_neighbourhood(window=1)

    with pytest.raises(InvalidArgumentValueError, match="`window` must be an integer in the range \\[2, inf\\), got"):
        lk.sorted_neighbourhood(window=0)


def test_sorted_neighbourhood_rejects_non_integer_window():
    """`window` must be a positive integer >= 2; None is a type error too
    (unlike fuzzy's topn, there is no absent state)."""
    for bad in (2.0, "2", True, None, []):
        with pytest.raises(InvalidArgumentTypeError, match="`window` must be an integer, got"):
            lk.sorted_neighbourhood(window=bad)


def test_sorted_neighbourhood_str_renders_window():
    assert "sorted_neighbourhood(" in str(lk.sorted_neighbourhood(window=3))
    assert "threshold=0.95" in str(lk.sorted_neighbourhood(window=3))
    assert "scorer='simple_ratio'" in str(lk.sorted_neighbourhood(window=3))
    assert "window=3" in str(lk.sorted_neighbourhood(window=3))
