"""Liken deduper unit tests"""

import math
from decimal import Decimal
from unittest.mock import Mock

import pyarrow as pa
import pytest
from rapidfuzz import fuzz

import liken as lk
from liken.dedupers.fuzzy import Fuzzy
from liken.exceptions import InvalidArgumentTypeError
from liken.exceptions import InvalidArgumentValueError


##############################
# fuzzy null partitioning   #
#############################


def test_fuzzy_nulls_pair_with_nulls_only(mock_df):
    """Missing values pair with each other, never with a value.

    fuzz.ratio("nathan", "nadia") is ~54.5, so the two values also pair at a
    0.5 threshold; the nulls form their own group either way.
    """
    mock_df.get_array = Mock(return_value=pa.array(["nathan", None, "nadia", None]))

    uf, n = lk.fuzzy(threshold=0.5).set_frame(mock_df).build_union_find("address", [])

    assert n == 4
    assert uf[1] == uf[3]  # the two nulls group
    assert uf[0] == uf[2]  # the values group at this threshold
    assert uf[0] != uf[1]  # a null never groups with a value


@pytest.mark.parametrize(
    "threshold, expected_pairs",
    [(math.nextafter(0.0, 1.0), [(0, 2), (1, 3)]), (0.55, [(1, 3)])],
    ids=["near-zero-threshold", "just-above-the-value-pair"],
)
def test_fuzzy_null_never_pairs_with_value(threshold, expected_pairs):
    """At any threshold a null pairs only with another null."""
    array = pa.array(["nathan", None, "nadia", None])

    pairs = sorted(lk.fuzzy(threshold=threshold)._gen_similarity_pairs(array))

    assert pairs == expected_pairs


def test_fuzzy_all_null_input_yields_star_pairs():
    """All-null input yields star-shaped null pairs without touching the scorer."""
    array = pa.array([None, None, None])

    assert list(lk.fuzzy(threshold=0.5)._gen_similarity_pairs(array)) == [(0, 1), (0, 2)]


def test_fuzzy_nan_and_null_pair():
    """NaN is missing and pairs with a null, never with a value."""
    array = pa.array([float("nan"), 1.0, None])

    assert list(lk.fuzzy(threshold=0.5)._gen_similarity_pairs(array)) == [(0, 2)]


def test_fuzzy_cdist_receives_no_missing_values(monkeypatch):
    """process.cdist scores non-missing values only."""
    from rapidfuzz import process

    from liken.core.missing import is_missing

    real_cdist = process.cdist

    def spy_cdist(queries, choices, **kwargs):
        assert all(not is_missing(value) for value in (*queries, *choices))
        return real_cdist(queries, choices, **kwargs)

    monkeypatch.setattr(process, "cdist", spy_cdist)

    array = pa.array(["nathan", None, "nadia", None])

    assert list(lk.fuzzy(threshold=0.95)._gen_similarity_pairs(array)) == [(1, 3)]


def test_fuzzy_unknown_scorer_raises_at_construction():
    """an unknown scorer is rejected at construction"""
    with pytest.raises(InvalidArgumentValueError, match="`scorer` must be one of"):
        Fuzzy(threshold=0.95, scorer="no_such_scorer")


def test_fuzzy_rejects_non_str_scorer():
    with pytest.raises(InvalidArgumentTypeError, match="`scorer` must be of type 'str', got 'int'"):
        lk.fuzzy(scorer=1)


def test_fuzzy_jaro_winkler_scorer_maps_onto_0_100_scale():
    """The registry wrapper scales normalized_similarity to the 0-100 scale.

    Raw `JaroWinkler.normalized_similarity` returns 0-1; the deduper
    compares scores against `100 * threshold`, so the registry value must
    return 100.0 for identical values and 0.0 for disjoint ones.
    """
    scorer = Fuzzy(threshold=0.95, scorer="jaro_winkler").get_scorer()

    assert scorer("abc", "abc") == 100.0
    assert scorer("abc", "xyz") == 0.0


def test_fuzzy_jaro_winkler_matches_real_pair_at_scaled_threshold():
    """A 0.905 pair matches at threshold 0.9 and not at 0.91.

    The fixture pair scores 0.905041 with JaroWinkler. A raw 0-1 score
    compared unconverted against `100 * threshold` (90, 91) would match
    neither, so the threshold=0.9 case fails if the scale conversion is
    dropped; the 0.91 case then separates a converted 90.5 from a
    threshold of 91.
    """
    s1 = "Calle Ancho, 12, 05688, Rioja, Navarra, Espana"
    s2 = "Calle Sueco, 56, 05688, Rioja, Navarra"

    pairs = list(lk.fuzzy(threshold=0.9, scorer="jaro_winkler")._gen_similarity_pairs(pa.array([s1, s2])))

    assert pairs == [(0, 1)]

    pairs = list(lk.fuzzy(threshold=0.91, scorer="jaro_winkler")._gen_similarity_pairs(pa.array([s1, s2])))

    assert pairs == []


def test_fuzzy_jaro_winkler_identical_pair_matches_high_threshold_mismatched_does_not():
    """Threshold is validated in the open interval (0, 1), so both probes stay in it."""
    deduper = lk.fuzzy(threshold=0.99, scorer="jaro_winkler")

    assert list(deduper._gen_similarity_pairs(pa.array(["abc", "abc"]))) == [(0, 1)]

    deduper = lk.fuzzy(threshold=0.01, scorer="jaro_winkler")

    assert list(deduper._gen_similarity_pairs(pa.array(["abc", "xyz"]))) == []


def test_fuzzy_default_scorer_is_unchanged():
    assert lk.fuzzy().get_scorer() is fuzz.ratio

    # the existing six scorers still construct
    for name in [
        "simple_ratio",
        "partial_ratio",
        "token_sort_ratio",
        "token_set_ratio",
        "weighted_ratio",
        "quick_ratio",
    ]:
        assert lk.fuzzy(scorer=name).get_scorer() is not None


###########################
# fuzzy topn acceptance  #
##########################

# WI050 (knn) mirrors these acceptance-semantics tests: keep the
# `test_fuzzy_topn_*` names and the pinned semantics identical.


def _stub_score_table(table):
    """A scorer stub driven by a value-keyed score table.

    Accepts `score_cutoff` as a kwarg: `process.extract` passes it. The
    stub returns the table score for the pair, or 0.0 for unlisted pairs.
    """

    def scorer(s1, s2, score_cutoff=None):
        return table.get((s1, s2), table.get((s2, s1), 0.0))

    return scorer


def test_fuzzy_topn_none_is_all_pairs():
    """`topn=None` (the default) is all-pairs fuzzy.

    With topn=1 the pair set on this fixture would differ: "abcf" is only
    the third-best partner of "abcd" and would be dropped.
    """
    array = pa.array(["abcd", "abce", "abcf", "zzzz"])

    default_pairs = sorted(lk.fuzzy(threshold=0.7)._gen_similarity_pairs(array))

    assert default_pairs == [(0, 1), (0, 2), (1, 2)]

    explicit_none_pairs = sorted(lk.fuzzy(threshold=0.7, topn=None)._gen_similarity_pairs(array))

    assert explicit_none_pairs == default_pairs


def test_fuzzy_accepts_topn_none():
    """`topn=None` constructs: None means the all-pairs default.

    Deliberately unlike tfidf, which rejects `topn=None` (its default 2 is
    an active value, not "absent").
    """
    lk.fuzzy(topn=None)


def test_fuzzy_topn_either_direction_emits_pair(monkeypatch):
    """A pair is emitted when either side holds the other in its k best.

    Score table: s(A,B)=80, s(A,C)=70, s(B,C)=60; topn=1, threshold 0.75.
    A's best is B (80), B's best is A (80): pair (0, 1). C's best is A
    (70, below threshold): no pair via C.

    At threshold 0.55, C's best (A at 70) clears the threshold while A's
    best is still B: pair (0, 2) survives through C's direction alone —
    the asymmetric case a one-direction-per-pair scan would miss.
    """
    scores = {("A", "B"): 80.0, ("A", "C"): 70.0, ("B", "C"): 60.0}

    deduper = lk.fuzzy(threshold=0.75, topn=1)
    monkeypatch.setattr(deduper, "get_scorer", lambda: _stub_score_table(scores))

    pairs = sorted(deduper._gen_similarity_pairs(pa.array(["A", "B", "C"])))

    assert pairs == [(0, 1)]

    deduper = lk.fuzzy(threshold=0.55, topn=1)
    monkeypatch.setattr(deduper, "get_scorer", lambda: _stub_score_table(scores))

    pairs = sorted(deduper._gen_similarity_pairs(pa.array(["A", "B", "C"])))

    assert pairs == [(0, 1), (0, 2)]


def test_fuzzy_topn_k_membership_is_per_row(monkeypatch):
    """Each row keeps its own k best partners.

    Scores: s(A,B)=80, s(A,C)=75, s(A,D)=70; s(B,C)=s(B,D)=s(C,D)=10.
    topn=1, threshold 0.6: A's best is B only; C's best is A (75) so
    (0, 2) survives via C's direction; D's best is A (70) so (0, 3)
    survives too. topn is per row: one row's k does not bound another
    row's partner count.
    """
    scores = {
        ("A", "B"): 80.0,
        ("A", "C"): 75.0,
        ("A", "D"): 70.0,
        ("B", "C"): 10.0,
        ("B", "D"): 10.0,
        ("C", "D"): 10.0,
    }

    deduper = lk.fuzzy(threshold=0.6, topn=1)
    monkeypatch.setattr(deduper, "get_scorer", lambda: _stub_score_table(scores))

    pairs = sorted(deduper._gen_similarity_pairs(pa.array(["A", "B", "C", "D"])))

    assert pairs == [(0, 1), (0, 2), (0, 3)]


@pytest.mark.parametrize(
    "drift, expected",
    [(0.0, [(0, 1)]), (-5e-10, [(0, 1)]), (-2e-9, [])],
    ids=["exact-at-threshold", "within-tolerance", "beyond-tolerance"],
)
def test_fuzzy_topn_boundary_drift(monkeypatch, drift, expected):
    """The top-k path shares the all-pairs boundary contract.

    The stub returns `100 * threshold + drift` for every pair. At exact
    boundary and within the 1e-9 tolerance the pair is emitted; beyond the
    tolerance nothing matches — and nothing below the band can appear in a
    top-k list, because extract's cutoff already filters it.
    """
    threshold = 0.75
    score = float(Decimal(str(threshold)) * 100) + drift

    deduper = lk.fuzzy(threshold=threshold, topn=1)
    monkeypatch.setattr(deduper, "get_scorer", lambda: lambda s1, s2, score_cutoff=None: score)

    pairs = list(deduper._gen_similarity_pairs(pa.array(["x", "y"])))

    assert pairs == expected


def test_fuzzy_topn_matches_a_real_score_exactly_at_the_threshold():
    """The native-scorer cutoff path matches exactly at the threshold.

    `fuzz.ratio("abcd", "abce")` is exactly 75.0: threshold 0.75 matches
    and 0.76 does not, through `process.extract`'s own cutoff comparison.
    """
    assert fuzz.ratio("abcd", "abce") == 75.0

    pairs = list(lk.fuzzy(threshold=0.75, topn=1)._gen_similarity_pairs(pa.array(["abcd", "abce"])))

    assert pairs == [(0, 1)]

    pairs = list(lk.fuzzy(threshold=0.76, topn=1)._gen_similarity_pairs(pa.array(["abcd", "abce"])))

    assert pairs == []


def test_fuzzy_topn_ties_broken_by_index_order():
    """Ties at rank k resolve by choice index, not by permutation invariance.

    Five identical values, topn=2, threshold 0.99: every score is 100, so
    each row's k best are the two lowest-index other rows. The undirected
    pair set is {(0,1), (0,2), (1,2), (0,3), (1,3), (0,4), (1,4)} — 7
    pairs. Reordering the input can change the result: the same input
    order gives the same output, nothing more (WI017 concern).
    """
    array = pa.array(["A"] * 5)

    pairs = sorted(lk.fuzzy(threshold=0.99, topn=2)._gen_similarity_pairs(array))

    assert pairs == [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)]


def test_fuzzy_topn_exceeding_partners_matches_all_pairs():
    """k >= n - 1 degenerates to all-pairs."""
    array = pa.array(["abcd", "abce", "abcf"])

    all_pairs = sorted(lk.fuzzy(threshold=0.7)._gen_similarity_pairs(array))
    topn_pairs = sorted(lk.fuzzy(threshold=0.7, topn=10)._gen_similarity_pairs(array))

    assert all_pairs == [(0, 1), (0, 2), (1, 2)]
    assert topn_pairs == all_pairs


def test_fuzzy_topn_jaro_winkler_matches_at_scaled_threshold():
    """The jaro_winkler wrapper works under the top-k path too.

    `process.extract` passes a real `score_cutoff` to the wrapper, which
    ignores it and returns the full 0-100 score; extract then compares
    that score against the cutoff itself. The 0.905 pair matches at
    threshold 0.9 and not at 0.91, same as the all-pairs path.
    """
    s1 = "Calle Ancho, 12, 05688, Rioja, Navarra, Espana"
    s2 = "Calle Sueco, 56, 05688, Rioja, Navarra"

    pairs = list(lk.fuzzy(threshold=0.9, scorer="jaro_winkler", topn=1)._gen_similarity_pairs(pa.array([s1, s2])))

    assert pairs == [(0, 1)]

    pairs = list(lk.fuzzy(threshold=0.91, scorer="jaro_winkler", topn=1)._gen_similarity_pairs(pa.array([s1, s2])))

    assert pairs == []


def test_fuzzy_topn_nulls_star_pair_outside_topk():
    """Missing values keep the star-shaped handling, outside the top-k path.

    fuzz.ratio("a", "ab") is ~66.7: no value pair at threshold 0.95. The
    null star pair (1, 3) is unaffected by topn.
    """
    array = pa.array(["a", None, "ab", None])

    pairs = sorted(lk.fuzzy(threshold=0.95, topn=1)._gen_similarity_pairs(array))

    assert pairs == [(1, 3)]


def test_fuzzy_topn_all_null_input_yields_star_pairs():
    array = pa.array([None, None, None])

    assert list(lk.fuzzy(threshold=0.5, topn=1)._gen_similarity_pairs(array)) == [(0, 1), (0, 2)]


def test_fuzzy_topn_extract_receives_no_missing_values(monkeypatch):
    """process.extract scores non-missing values only."""
    from rapidfuzz import process

    from liken.core.missing import is_missing

    real_extract = process.extract

    def spy_extract(query, choices, **kwargs):
        assert all(not is_missing(value) for value in (query, *choices))
        return real_extract(query, choices, **kwargs)

    monkeypatch.setattr(process, "extract", spy_extract)

    array = pa.array(["nathan", None, "nadia", None])

    assert list(lk.fuzzy(threshold=0.95, topn=1)._gen_similarity_pairs(array)) == [(1, 3)]


def test_fuzzy_topn_transitivity_through_union_find(mock_df, monkeypatch):
    """Merged pairs chain transitively through union-find.

    Scores: s(A,B)=80, s(A,C)=60, s(B,C)=75; topn=1, threshold 0.7.
    Pairs (A,B) and (B,C) qualify; A-C never does. Union-find still merges
    all three through B — the docstring's transitivity note in action.
    """
    scores = {("A", "B"): 80.0, ("A", "C"): 60.0, ("B", "C"): 75.0}

    deduper = lk.fuzzy(threshold=0.7, topn=1)
    monkeypatch.setattr(deduper, "get_scorer", lambda: _stub_score_table(scores))

    mock_df.get_array = Mock(return_value=pa.array(["A", "B", "C"]))

    uf, n = deduper.set_frame(mock_df).build_union_find("address", [])

    assert n == 3
    assert uf[0] == uf[1] == uf[2]


def test_fuzzy_topn_calls_extract_once_per_present_row(monkeypatch):
    """The mechanism is candidate-limited per row, not a trimmed full matrix.

    `process.extract` runs exactly once per present row, each call with
    `limit=k+1` and the full present-value list as choices.
    """
    from rapidfuzz import process

    real_extract = process.extract
    calls = []

    def spy_extract(query, choices, **kwargs):
        calls.append({"limit": kwargs.get("limit"), "choices_len": len(choices)})
        return real_extract(query, choices, **kwargs)

    monkeypatch.setattr(process, "extract", spy_extract)

    array = pa.array(["abcd", "abce", "abcf", None])

    pairs = list(lk.fuzzy(threshold=0.7, topn=2)._gen_similarity_pairs(array))

    assert pairs == [(0, 1), (0, 2), (1, 2)]
    assert len(calls) == 3  # one per present row
    assert all(call["limit"] == 3 for call in calls)  # k + 1
    assert all(call["choices_len"] == 3 for call in calls)  # full present list


@pytest.mark.parametrize("topn", [0, -1])
def test_fuzzy_rejects_non_positive_topn(topn):
    """`topn` below 1 is refused at construction: an unvalidated topn reaches
    the native library and can abort the process."""
    with pytest.raises(InvalidArgumentValueError, match="`topn` must be an integer in the range \\[1, inf\\), got"):
        lk.fuzzy(topn=topn)


@pytest.mark.parametrize("topn", [2.0, "2", True, []])
def test_fuzzy_rejects_non_integer_topn(topn):
    """`topn` must be a positive integer or None.

    tfidf rejects None (its default 2 is an active value); fuzzy
    deliberately accepts it — None is the all-pairs default here.
    """
    with pytest.raises(InvalidArgumentTypeError, match="`topn` must be an integer, got"):
        lk.fuzzy(topn=topn)


def test_fuzzy_topn_str_repr_omitted_when_none():
    """The default repr stays `fuzzy(threshold=0.95)`; a set topn renders."""
    assert "topn" not in str(lk.fuzzy())
    assert "topn=5" in str(lk.fuzzy(topn=5))


##################################


@pytest.mark.parametrize(
    "threshold",
    [i / 100 for i in range(1, 100)],
    ids=[f"{i / 100:.2f}" for i in range(1, 100)],
)
def test_fuzzy_matches_score_exactly_at_every_two_decimal_threshold(threshold, monkeypatch):
    """A pair whose score is exactly `100 * threshold` matches, for every two-decimal threshold.

    The stubbed scorer returns the exact product as a float. The drifted
    `100 * threshold` (8 of the 99 two-decimal thresholds round away from
    it, 5 of them upward) must not reject a pair sitting exactly on the
    boundary.
    """
    exact_score = float(Decimal(str(threshold)) * 100)
    deduper = lk.fuzzy(threshold=threshold)
    monkeypatch.setattr(deduper, "get_scorer", lambda: lambda s1, s2, **kwargs: exact_score)

    pairs = list(deduper._gen_similarity_pairs(pa.array(["x", "y"])))

    assert pairs == [(0, 1)]


def test_fuzzy_does_not_match_below_the_threshold(monkeypatch):
    """A score more than the tolerance below the boundary never matches."""
    deduper = lk.fuzzy(threshold=0.95)
    monkeypatch.setattr(deduper, "get_scorer", lambda: lambda s1, s2, **kwargs: 94.0)

    assert list(deduper._gen_similarity_pairs(pa.array(["x", "y"]))) == []


def test_fuzzy_matches_a_real_score_exactly_at_the_threshold():
    """`fuzz.ratio("abcd", "abce")` is exactly 75.0, so threshold 0.75 matches and 0.76 does not."""
    assert fuzz.ratio("abcd", "abce") == 75.0

    pairs = list(lk.fuzzy(threshold=0.75)._gen_similarity_pairs(pa.array(["abcd", "abce"])))

    assert pairs == [(0, 1)]

    pairs = list(lk.fuzzy(threshold=0.76)._gen_similarity_pairs(pa.array(["abcd", "abce"])))

    assert pairs == []
