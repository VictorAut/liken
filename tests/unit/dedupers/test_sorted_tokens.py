"""Liken deduper unit tests"""

import random
from unittest.mock import Mock

import pyarrow as pa
import pytest

import liken as lk
from liken.exceptions import InvalidArgumentTypeError


def test_date_range_empty_array_yields_no_pairs():
    deduper = lk.date_range(format="%Y-%m-%d")
    deduper.validate("date")

    assert list(deduper._gen_similarity_pairs(pa.array([], type=pa.string()))) == []


###################
#  sorted_tokens  #
###################


@pytest.mark.parametrize(
    "values",
    [
        ["John Smith", "Smith John"],
        ["John Smith", "John Smith"],
        ["  John   Smith  ", "John Smith"],
        ["John\tSmith", "John Smith"],
        ["Mary Ann Lee", "Lee Ann Mary"],
    ],
    ids=["swap", "identical", "padded", "tab", "three-token-swap"],
)
def test_sorted_tokens_groups_word_order_swaps(values):
    deduper = lk.sorted_tokens()
    deduper.validate("name")

    pairs = list(deduper._gen_similarity_pairs(pa.array(values)))

    assert pairs == [(0, 1)]


@pytest.mark.parametrize(
    "values",
    [
        ["John Smith", "Jon Smith"],
        ["John Smith", "John Smithy"],
        ["John Smith", "John  Smithy"],
        ["Main St.", "Main, St."],
    ],
    ids=["different-word", "different-token", "same-multiset", "punctuation-placement"],
)
def test_sorted_tokens_never_groups_different_token_multisets(values):
    deduper = lk.sorted_tokens()
    deduper.validate("name")

    pairs = list(deduper._gen_similarity_pairs(pa.array(values)))

    assert pairs == []


def test_sorted_tokens_single_token_values_behave_as_exact():
    """A single token sorts to itself: matching degenerates to equality."""
    deduper = lk.sorted_tokens()
    deduper.validate("city")

    pairs = list(deduper._gen_similarity_pairs(pa.array(["london", "london", "london2"])))

    assert pairs == [(0, 1)]


def test_sorted_tokens_transitive_grouping(mock_df):
    """Values sharing a sorted-token key merge through union-find; distinct
    keys stay apart."""
    mock_df.get_array = Mock(return_value=pa.array(["John Smith", "Smith John", "John Smith", "Peter Paul"]))

    uf, n = lk.sorted_tokens().set_frame(mock_df).build_union_find("name", [])

    assert n == 4
    assert uf[0] == uf[1] == uf[2]
    assert uf[3] != uf[0]


@pytest.mark.parametrize(
    "values",
    [
        ["John Smith", "", "Jane Doe", ""],
        ["John Smith", "   ", "Jane Doe"],
    ],
    ids=["empty", "whitespace-only"],
)
def test_sorted_tokens_zero_token_values_never_match(values):
    """An empty or whitespace-only string carries no tokens: it matches
    nothing on the positive path, not even its twin."""
    deduper = lk.sorted_tokens()
    deduper.validate("name")

    pairs = list(deduper._gen_similarity_pairs(pa.array(values)))

    assert pairs == []


def test_sorted_tokens_nulls_pair_with_nulls_only():
    deduper = lk.sorted_tokens()
    deduper.validate("name")

    pairs = list(deduper._gen_similarity_pairs(pa.array(["John Smith", None, None, "Jane Doe"])))

    assert pairs == [(1, 2)]


@pytest.mark.parametrize(
    "values",
    [
        ["John Smith", "", "   ", None, "Smith John"],
        ["John Smith", "", None, ""],
    ],
    ids=["both-zero-token-classes", "two-empties"],
)
def test_sorted_tokens_negated_groups_zero_token_values(values):
    """~sorted_tokens groups zero-token values (empty and whitespace-only);
    values with tokens and nulls match nothing under negation."""
    deduper = ~lk.sorted_tokens()
    deduper.validate("name")

    pairs = list(deduper._gen_similarity_pairs(pa.array(values)))

    zero_token = [i for i, value in enumerate(values) if isinstance(value, str) and not value.split()]
    assert len(zero_token) >= 2
    assert all(a in zero_token and b in zero_token for a, b in pairs)
    # every pair of zero-token values pairs
    expected = {(a, b) for i, a in enumerate(zero_token) for b in zero_token[i + 1 :]}
    assert set(pairs) == expected


def test_sorted_tokens_negated_never_matches_nulls():
    deduper = ~lk.sorted_tokens()
    deduper.validate("name")

    pairs = list(deduper._gen_similarity_pairs(pa.array([None, "John Smith"])))

    assert pairs == []


def test_sorted_tokens_rejects_non_string_arrays_at_execution():
    deduper = lk.sorted_tokens()
    deduper.validate("n")

    with pytest.raises(InvalidArgumentTypeError, match="string"):
        list(deduper._gen_similarity_pairs(pa.array([1, 2])))


def test_sorted_tokens_boolean_array_rejected_at_execution():
    deduper = lk.sorted_tokens()
    deduper.validate("flag")

    with pytest.raises(InvalidArgumentTypeError, match="string"):
        list(deduper._gen_similarity_pairs(pa.array([True, False])))


def test_sorted_tokens_negated_rejects_non_string_array():
    """The negated path type-checks the column too: without the check a
    non-string column would fail `_matches` per value and `~sorted_tokens`
    would silently star-group every row."""
    deduper = ~lk.sorted_tokens()
    deduper.validate("n")

    with pytest.raises(InvalidArgumentTypeError, match="string"):
        list(deduper._gen_similarity_pairs(pa.array([1, 2])))


def test_sorted_tokens_empty_array_yields_no_pairs():
    deduper = lk.sorted_tokens()
    deduper.validate("name")

    assert list(deduper._gen_similarity_pairs(pa.array([], type=pa.string()))) == []


def test_sorted_tokens_requires_single_string_column():
    with pytest.raises(InvalidArgumentTypeError, match="`columns` must be of type 'str'"):
        lk.sorted_tokens().validate(("a", "b"))


def test_sorted_tokens_permutation_invariance(mock_df):
    """The same multiset of values in different row orders yields the same
    grouping up to relabelling (WI017 concern): sort keys are token strings
    in code-point order, no locale collation. Comparison is on partitions
    (frozensets of member indices), not on raw union-find roots."""
    values = ["John Smith", "Smith John", "Mary Ann Lee", "Lee Mary Ann", "Ann Lee Mary", "Peter Paul"]

    def grouping(order):
        mock_df.get_array = Mock(return_value=pa.array([values[i] for i in order]))
        uf, _ = lk.sorted_tokens().set_frame(mock_df).build_union_find("name", [])
        members: dict[object, set[int]] = {}
        for i in range(len(order)):
            members.setdefault(uf[i], set()).add(i)
        return {frozenset(group) for group in members.values()}

    rng = random.Random(7)
    reference = grouping(list(range(len(values))))
    for _ in range(20):
        order = list(range(len(values)))
        rng.shuffle(order)
        shuffled = grouping(order)
        # shuffle[i] holds values[order[i]]; map each shuffled group to the
        # values it holds, so the two partitions compare directly
        relabelled = {frozenset(order[i] for i in group) for group in shuffled}
        assert relabelled == reference


def test_sorted_tokens_pair_emission_is_deterministic():
    deduper = lk.sorted_tokens()
    deduper.validate("name")

    values = ["John Smith", "Smith John", "Mary Ann Lee", "Lee Mary Ann"]

    first = list(deduper._gen_similarity_pairs(pa.array(values)))
    second = list(deduper._gen_similarity_pairs(pa.array(values)))

    assert first == second


def test_sorted_tokens_str_renders_no_arguments():
    assert "sorted_tokens()" in str(lk.sorted_tokens())
