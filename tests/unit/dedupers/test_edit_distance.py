"""Liken deduper unit tests"""

from decimal import Decimal

import numpy as np
import pyarrow as pa
import pytest

import liken as lk
from liken.dedupers.edit_distance import EditDistance
from liken.exceptions import InvalidArgumentTypeError
from liken.exceptions import InvalidArgumentValueError


##################
# edit_distance #
##################


def test_edit_distance_default_max_distance_is_two():
    deduper = lk.edit_distance()

    assert deduper._max_distance == 2


def test_edit_distance_matches_pairs_at_inclusive_bound():
    """Values exactly max_distance edits apart are still matched."""
    deduper = EditDistance(max_distance=2)

    pairs = list(deduper._gen_similarity_pairs(pa.array(["abcde", "abcde", "abcxy"])))

    assert (0, 2) in pairs


def test_edit_distance_rejects_pairs_just_above_bound():
    """One edit beyond the bound is not matched."""
    deduper = EditDistance(max_distance=2)

    pairs = list(deduper._gen_similarity_pairs(pa.array(["abcde", "abcde", "abcxyz"])))

    # only the identical pair matches; "abcde"/"abcxyz" are 3 edits apart
    assert pairs == [(0, 1)]


def test_edit_distance_rejects_long_values_with_many_edits():
    """The absolute bound holds regardless of string length."""
    deduper = EditDistance(max_distance=2)

    pairs = list(deduper._gen_similarity_pairs(pa.array(["a" * 20 + "b", "a" * 20 + "b", "c" * 20 + "b"])))

    # only the identical pair matches; the third value is 20 edits away
    assert pairs == [(0, 1)]


def test_edit_distance_negative_max_distance_raises_value_error():
    with pytest.raises(InvalidArgumentValueError, match="`max_distance` must be an integer in the range \\[0, inf\\)"):
        lk.edit_distance(max_distance=-1)


@pytest.mark.parametrize("value", [1.5, "2", True, np.int64(2)])
def test_edit_distance_non_integer_max_distance_raises_type_error(value):
    with pytest.raises(InvalidArgumentTypeError, match="`max_distance` must be an integer, got"):
        lk.edit_distance(max_distance=value)


def test_edit_distance_default_metric_is_levenshtein():
    assert lk.edit_distance()._metric == "levenshtein"


@pytest.mark.parametrize("metric", ["damerau_levenshtein", "jaro_winkler", "indel", "hamming"])
def test_edit_distance_accepts_each_non_default_metric(metric):
    assert lk.edit_distance(max_distance=0.5, metric=metric)._metric == metric


def test_edit_distance_unknown_metric_raises_value_error():
    with pytest.raises(InvalidArgumentValueError, match="`metric` must be one of"):
        lk.edit_distance(metric="levenshteinn")


def test_edit_distance_unknown_metric_error_names_all_options():
    with pytest.raises(InvalidArgumentValueError) as exc_info:
        lk.edit_distance(metric="nope")

    message = str(exc_info.value)
    for name in ("levenshtein", "damerau_levenshtein", "jaro_winkler", "indel", "hamming"):
        assert name in message


def test_edit_distance_non_str_metric_raises_type_error():
    with pytest.raises(InvalidArgumentTypeError, match="`metric` must be of type 'str', got 'int'"):
        lk.edit_distance(metric=1)


def test_edit_distance_normalized_metric_rejects_default_max_distance():
    """The default absolute bound 2 is out of range for the normalized contract."""
    with pytest.raises(InvalidArgumentValueError, match="`max_distance` must be a real number in the range \\[0, 1\\]"):
        lk.edit_distance(metric="jaro_winkler")


def test_edit_distance_normalized_metric_accepts_float_in_range():
    assert lk.edit_distance(max_distance=0.25, metric="indel")._max_distance == 0.25


def test_edit_distance_normalized_metric_accepts_integer_zero_and_one():
    """Ints 0 and 1 sit on the inclusive bounds and are real numbers."""
    assert lk.edit_distance(max_distance=0, metric="hamming")._max_distance == 0
    assert lk.edit_distance(max_distance=1, metric="hamming")._max_distance == 1


@pytest.mark.parametrize("value", [1.5, -0.1, True, float("nan"), "0.5", np.float32(0.5)])
def test_edit_distance_normalized_metric_rejects_bad_max_distance(value):
    with pytest.raises((InvalidArgumentTypeError, InvalidArgumentValueError)):
        lk.edit_distance(max_distance=value, metric="indel")


def test_edit_distance_levenshtein_rejects_float_max_distance():
    """The absolute edit-count contract keeps its integer-only typing."""
    with pytest.raises(InvalidArgumentTypeError, match="`max_distance` must be an integer"):
        lk.edit_distance(max_distance=0.5)


#####################
# metric behaviour  #
#####################


def test_edit_distance_damerau_levenshtein_matches_transposition():
    """Damerau-Levenshtein counts a transposition as one edit: nd('abcd','abdc') = 0.25."""
    deduper = EditDistance(max_distance=0.25, metric="damerau_levenshtein")

    pairs = list(deduper._gen_similarity_pairs(pa.array(["abcd", "abdc"])))

    assert pairs == [(0, 1)]

    # the same shape under plain levenshtein is 2 edits: rejected at 1, matched at 2
    assert list(lk.edit_distance(max_distance=1)._gen_similarity_pairs(pa.array(["abcd", "abdc"]))) == []
    assert list(lk.edit_distance(max_distance=2)._gen_similarity_pairs(pa.array(["abcd", "abdc"]))) == [(0, 1)]


def test_edit_distance_jaro_winkler_weights_shared_prefixes():
    """A same-shape single-edit pair scores closer when it shares a prefix.

    nd('abcdx','abcdy') = 0.08 with the shared prefix; nd('abcdx','xbcda')
    = 0.267 without it. A bound between the two separates them.
    """
    deduper = EditDistance(max_distance=0.15, metric="jaro_winkler")

    pairs = list(deduper._gen_similarity_pairs(pa.array(["abcdx", "abcdy", "xbcda"])))

    assert pairs == [(0, 1)]


def test_edit_distance_indel_matches_pure_insertion():
    """Indel nd('abc','abcd') = 0.25; a substitution would cost double."""
    deduper = EditDistance(max_distance=0.25, metric="indel")

    pairs = list(deduper._gen_similarity_pairs(pa.array(["abc", "abcd"])))

    assert pairs == [(0, 1)]


def test_edit_distance_hamming_matches_substitution_pair():
    """Hamming nd('karolin','kerstin') = 3/7 ≈ 0.4286."""
    deduper = EditDistance(max_distance=0.45, metric="hamming")

    pairs = list(deduper._gen_similarity_pairs(pa.array(["karolin", "kerstin"])))

    assert pairs == [(0, 1)]


def test_edit_distance_hamming_pads_unequal_lengths():
    """Hamming pads unequal-length values through cdist (pad=True default).

    Verified against rapidfuzz 3.14.3: nd('abc','abcde') = 0.4. Documented
    behaviour, pinned here as documentation-by-test.
    """
    deduper = EditDistance(max_distance=0.4, metric="hamming")

    pairs = list(deduper._gen_similarity_pairs(pa.array(["abc", "abcde"])))

    assert pairs == [(0, 1)]


@pytest.mark.parametrize("metric", ["damerau_levenshtein", "indel", "hamming"])
def test_edit_distance_normalized_metric_inclusive_boundary(metric):
    """A pair exactly at `max_distance` matches; one beyond the tolerance does not.

    Exact-representable bounds (0.25 on len-4 strings) so the at-bound
    expectation does not depend on the drift guard. A bound one step below
    rejects: 0.25 exceeds it by more than the guard.
    """
    at_bound = EditDistance(max_distance=0.25, metric=metric)
    below_bound = EditDistance(max_distance=0.24, metric=metric)

    values = ["abcd", "abce"]  # one substitution over length 4: nd = 0.25

    assert list(at_bound._gen_similarity_pairs(pa.array(values))) == [(0, 1)]
    assert list(below_bound._gen_similarity_pairs(pa.array(values))) == []


def test_edit_distance_jaro_winkler_rejects_pair_above_bound():
    """A pair whose nd is clearly above the bound never matches."""
    deduper = EditDistance(max_distance=0.25, metric="jaro_winkler")

    # nd('abcdx','xbcda') = 0.2667, no shared prefix
    assert list(deduper._gen_similarity_pairs(pa.array(["abcdx", "xbcda"]))) == []


def test_edit_distance_jaro_winkler_matches_at_stubbed_exact_bound(monkeypatch):
    """A computed distance exactly at `max_distance` matches (guard direction).

    rapidfuzz's JW normalized distance can drift below the true value, so a
    real boundary pair is admitted from below; the stub pins the at-bound
    admission itself.
    """
    deduper = EditDistance(max_distance=0.11666666666666667, metric="jaro_winkler")
    monkeypatch.setattr(
        "liken.dedupers.edit_distance.process.cdist",
        lambda queries, choices, **kwargs: [[0.11666666666666667]],
    )

    assert list(deduper._gen_similarity_pairs(pa.array(["abcd", "abce"]))) == [(0, 1)]


@pytest.mark.parametrize("metric", ["levenshtein", "damerau_levenshtein", "jaro_winkler", "indel", "hamming"])
def test_edit_distance_nulls_pair_with_nulls_only_for_every_metric(metric):
    """Missing values pair only with each other, under every metric."""
    max_distance = 2 if metric == "levenshtein" else 0.5
    deduper = EditDistance(max_distance=max_distance, metric=metric)

    pairs = sorted(deduper._gen_similarity_pairs(pa.array(["n", None, None, "na"])))

    assert pairs == [(0, 3), (1, 2)]


@pytest.mark.parametrize(
    "max_distance",
    [i / 100 for i in range(1, 100)],
    ids=[f"{i / 100:.2f}" for i in range(1, 100)],
)
def test_edit_distance_normalized_metric_matches_exactly_at_every_two_decimal_bound(max_distance, monkeypatch):
    """A pair whose distance is exactly `max_distance` matches, for every two-decimal bound.

    The stubbed scorer returns the bound exactly, bit-identical to
    `max_distance`, so the comparison is exact for all 99 bounds: the
    at-bound pair must match with no drift in play.
    """
    exact_distance = float(Decimal(str(max_distance)))
    deduper = EditDistance(max_distance=max_distance, metric="indel")
    monkeypatch.setattr(
        "liken.dedupers.edit_distance.process.cdist",
        lambda queries, choices, **kwargs: [[exact_distance]],
    )

    pairs = list(deduper._gen_similarity_pairs(pa.array(["x", "y"])))

    assert pairs == [(0, 1)]


@pytest.mark.parametrize("drift", [5e-10, 2e-9], ids=["within_tolerance", "beyond_tolerance"])
def test_edit_distance_normalized_metric_tolerance_guard(drift, monkeypatch):
    """A distance within the tolerance above `max_distance` matches; beyond it does not.

    Pins the guard's widening behaviour itself: `max_distance + 5e-10`
    (inside `_BOUNDARY_TOLERANCE` = 1e-9) is admitted,
    `max_distance + 2e-9` is rejected.
    """
    deduper = EditDistance(max_distance=0.5, metric="indel")
    monkeypatch.setattr(
        "liken.dedupers.edit_distance.process.cdist",
        lambda queries, choices, **kwargs: [[0.5 + drift]],
    )

    pairs = list(deduper._gen_similarity_pairs(pa.array(["x", "y"])))

    assert pairs == ([(0, 1)] if drift == 5e-10 else [])


def test_edit_distance_normalized_metric_does_not_match_above_the_bound(monkeypatch):
    """A distance more than the tolerance above the boundary never matches."""
    deduper = EditDistance(max_distance=0.5, metric="indel")
    monkeypatch.setattr(
        "liken.dedupers.edit_distance.process.cdist",
        lambda queries, choices, **kwargs: [[0.51]],
    )

    assert list(deduper._gen_similarity_pairs(pa.array(["x", "y"]))) == []


def test_edit_distance_requires_single_string_column():
    with pytest.raises(InvalidArgumentTypeError):
        EditDistance().validate(("a", "b"))


def test_edit_distance_nulls_pair_with_nulls_only():
    """Missing values pair only with each other; a null never pairs with a value.

    Recomputed from the null contract, replacing the placeholder-guard test:
    nulls never reach the scorer; Levenshtein("n", "na") is 1, so the two
    values pair at max_distance=2 as ordinary values.
    """
    deduper = EditDistance(max_distance=2)

    pairs = sorted(deduper._gen_similarity_pairs(pa.array(["n", None, None, "na"])))

    assert pairs == [(0, 3), (1, 2)]


def test_edit_distance_literal_na_is_an_ordinary_value():
    """Literal "na" values pair as values; a lone null pairs with nothing.

    Recomputed from the null contract: Levenshtein("na", "") is 2, so the
    empty string pairs with both "na" literals at max_distance=2.
    """
    deduper = EditDistance(max_distance=2)

    pairs = sorted(deduper._gen_similarity_pairs(pa.array(["na", None, "na", ""])))

    assert pairs == [(0, 2), (0, 3), (2, 3)]
