"""Liken deduper unit tests"""

import math
from unittest.mock import Mock

import pyarrow as pa
import pytest

import liken as lk
from liken.exceptions import InvalidArgumentTypeError
from liken.exceptions import InvalidArgumentValueError


###################
#  token_overlap  #
###################


def test_token_overlap_containment_matches_where_jaccard_would_not():
    """The spec's headline case: a 2-member set fully contained in a 4-member set.

    Containment is 2/2 = 1.0, so it matches at threshold 0.95 where the
    jaccard ratio would be 2/4 = 0.5.
    """
    table = pa.table({"x": ["a", "a"], "y": ["b", "b"], "z": [None, "c"], "w": [None, "d"]})

    assert list(lk.token_overlap(threshold=0.95)._gen_similarity_pairs(table)) == [(0, 1)]
    assert list(lk.jaccard(threshold=0.95)._gen_similarity_pairs(table)) == []


def test_token_overlap_subset_in_the_later_row_also_matches():
    """The denominator is min(|A|,|B|) regardless of which row holds the subset.

    The headline case with the rows swapped: a regression using
    len(sets[idx]) as the denominator would score this pair 2/4 and
    reject it at 0.95.
    """
    table = pa.table({"x": ["a", "a"], "y": ["b", "b"], "z": ["c", None], "w": ["d", None]})

    assert list(lk.token_overlap(threshold=0.95)._gen_similarity_pairs(table)) == [(0, 1)]


def test_token_overlap_matches_a_pair_exactly_at_the_threshold():
    """A pair sharing 1 of 2 members of the smaller set has containment exactly 0.5."""
    table = pa.table({"x": ["a", "a"], "y": ["b", "c"], "z": [None, "d"], "w": [None, "e"]})

    pairs = list(lk.token_overlap(threshold=0.5)._gen_similarity_pairs(table))

    assert pairs == [(0, 1)]


def test_token_overlap_does_not_match_below_the_threshold():
    table = pa.table({"x": ["a", "a"], "y": ["b", "c"], "z": [None, "d"], "w": [None, "e"]})

    pairs = list(lk.token_overlap(threshold=math.nextafter(0.5, math.inf))._gen_similarity_pairs(table))

    assert pairs == []


def test_token_overlap_matches_in_either_direction():
    """Either direction clearing the threshold emits the pair, once, in index order.

    Row 0's set {a} is fully contained in row 1's {a, b, c}: direction
    0->1 scores 1/1 = 1.0 even though direction 1->0 scores 1/3.
    """
    table = pa.table({"x": ["a", "a"], "y": [None, "b"], "z": [None, "c"]})

    pairs = list(lk.token_overlap(threshold=0.95)._gen_similarity_pairs(table))

    assert pairs == [(0, 1)]


def test_token_overlap_two_all_null_rows_do_not_match():
    """The empty-intersection short-circuit: two rows with no members match nothing."""
    table = pa.table({"x": [None, None], "y": [None, None]})

    assert list(lk.token_overlap(threshold=0.01)._gen_similarity_pairs(table)) == []


def test_token_overlap_one_all_null_row_does_not_match_a_populated_row():
    """An all-null row's set is empty, so its intersection with anything is empty."""
    table = pa.table({"x": [None, "a"], "y": [None, "b"]})

    assert list(lk.token_overlap(threshold=0.01)._gen_similarity_pairs(table)) == []


def test_token_overlap_excludes_none_members_from_both_sets():
    """A None member neither matches a value nor counts toward min(|A|, |B|).

    Dropping the None members from both rows leaves {a, b} against
    {a, b, c}: containment 2/2 = 1.0, so the pair matches at 0.95.
    """
    table = pa.table({"x": ["a", "a"], "y": ["b", "b"], "z": [None, "c"]})

    assert list(lk.token_overlap(threshold=0.95)._gen_similarity_pairs(table)) == [(0, 1)]


def test_token_overlap_is_not_tokenized():
    """Row sets are built from cell values as they are, like jaccard: "a b" and "ab" differ."""
    table = pa.table({"x": ["a b", "ab"]})

    assert list(lk.token_overlap(threshold=0.01)._gen_similarity_pairs(table)) == []


def test_token_overlap_groups_transitively_through_union_find(mock_df):
    """A->B and B->C each clear the threshold, so all three merge into one group.

    Row sets: {a} vs {a, b} is containment 1.0; {a, b} vs {a, b, c} is
    containment 1.0; {a} vs {a, b, c} is also 1.0.
    """
    mock_df.get_array = Mock(
        return_value=pa.table(
            {
                "x": ["a", "a", "a"],
                "y": [None, "b", "b"],
                "z": [None, None, "c"],
            }
        )
    )

    uf, n = lk.token_overlap(threshold=0.95).set_frame(mock_df).build_union_find(("x", "y", "z"), [])

    assert n == 3
    assert uf[0] == uf[1] == uf[2]


def test_token_overlap_requires_compound_tuple_columns():
    with pytest.raises(InvalidArgumentTypeError):
        lk.token_overlap().validate("name")

    with pytest.raises(InvalidArgumentTypeError):
        lk.token_overlap().validate(("name", 3))


@pytest.mark.parametrize("threshold", ["0.5", None, 0.0, 1.0, -0.5])
def test_token_overlap_rejects_thresholds_outside_the_exclusive_range(threshold):
    with pytest.raises((InvalidArgumentTypeError, InvalidArgumentValueError)):
        lk.token_overlap(threshold=threshold)


def test_token_overlap_str_renders_threshold():
    assert "token_overlap(" in str(lk.token_overlap(threshold=0.75))
    assert "threshold=0.75" in str(lk.token_overlap(threshold=0.75))
