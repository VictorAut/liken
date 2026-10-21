"""Liken deduper unit tests"""

import pyarrow as pa
import pytest

import liken as lk
from liken.exceptions import InvalidArgumentTypeError
from liken.exceptions import InvalidArgumentValueError


################
#  date_range  #
################


def test_date_range_requires_format():
    """`format` has no default: the deduper never guesses a date format."""
    with pytest.raises(TypeError):
        lk.date_range()


@pytest.mark.parametrize(
    "format",
    [None, 5, ["%Y-%m-%d"], True],
    ids=["none", "int", "list", "bool"],
)
def test_date_range_rejects_non_str_format(format):
    with pytest.raises(InvalidArgumentTypeError, match="`format` must be of type 'str', got"):
        lk.date_range(format=format)


def test_date_range_rejects_empty_format():
    with pytest.raises(InvalidArgumentValueError, match="`format` must be a non-empty string"):
        lk.date_range(format="")


@pytest.mark.parametrize(
    "window",
    [1.0, "1", True, None],
    ids=["float", "str", "bool", "none"],
)
def test_date_range_rejects_non_integer_window(window):
    with pytest.raises(InvalidArgumentTypeError, match="`window` must be an integer"):
        lk.date_range(format="%Y-%m-%d", window=window)


def test_date_range_rejects_negative_window():
    with pytest.raises(InvalidArgumentValueError, match="`window` must be an integer in the range"):
        lk.date_range(format="%Y-%m-%d", window=-1)


@pytest.mark.parametrize(
    "window",
    [0, 1, 30],
    ids=["zero", "one", "thirty"],
)
def test_date_range_accepts_valid_window(window):
    assert lk.date_range(format="%Y-%m-%d", window=window)._window == window


def test_date_range_default_window_is_zero():
    assert lk.date_range(format="%Y-%m-%d")._window == 0


def test_date_range_boundary_crossing_bucket_merge():
    """Jan 5 and Jan 7 at window=2 merge even though plain date-key
    bucketing would separate them; this pins the window+1 cell scheme."""
    deduper = lk.date_range(format="%Y-%m-%d", window=2)
    deduper.validate("date")

    pairs = list(deduper._gen_similarity_pairs(pa.array(["2024-01-05", "2024-01-07"])))

    assert pairs == [(0, 1)]


def test_date_range_chain_at_window_one_emits_adjacent_pairs_only():
    """Jan 5, 6, 7 at window=1: adjacent pairs match, Jan 5-Jan 7 does not."""
    deduper = lk.date_range(format="%Y-%m-%d", window=1)
    deduper.validate("date")

    pairs = sorted(deduper._gen_similarity_pairs(pa.array(["2024-01-05", "2024-01-06", "2024-01-07"])))

    assert pairs == [(0, 1), (1, 2)]


def test_date_range_dense_two_dates_at_window_zero():
    """Two dates 1 day apart at window=0 emit only the within-day stars.

    The neighbour join previously cross-multiplied the two cells (24.7s at
    n=30k for zero cross-cell pairs); the two-pointer scan visits each
    side once. Same two dates one day apart at window=1 merge everything.
    """
    values = ["2024-01-05"] * 3 + ["2024-01-06"] * 3

    zero = lk.date_range(format="%Y-%m-%d")
    zero.validate("date")
    assert sorted(zero._gen_similarity_pairs(pa.array(values))) == [(0, 1), (0, 2), (3, 4), (3, 5)]

    one = lk.date_range(format="%Y-%m-%d", window=1)
    one.validate("date")
    pairs = sorted(one._gen_similarity_pairs(pa.array(values)))

    # The within-day pairs are star-shaped (predicate convention); the
    # cross-day join emits every matching pair. Grouping: all six merge.
    assert pairs == [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 3),
        (2, 4),
        (2, 5),
        (3, 4),
        (3, 5),
    ]


def test_date_range_dense_mixed_dates_cross_cell():
    """Dense data over several days: the two-pointer neighbour join emits
    every matching cross-cell pair; within-cell emission is star-shaped
    from the cell's first index (predicate convention). The merged groups
    equal the brute-force within-window groups; pair shape is
    cell-alignment dependent."""
    values = [f"2024-01-{day:02d}" for day in (5, 5, 6, 6, 7, 7, 9, 10)]

    deduper = lk.date_range(format="%Y-%m-%d", window=1)
    deduper.validate("date")

    pairs = sorted(deduper._gen_similarity_pairs(pa.array(values)))

    assert pairs == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
        (2, 4),
        (2, 5),
        (6, 7),
    ]


def test_date_range_pre_epoch_dates_match_within_window():
    """Dates before 1970-01-01 pair correctly: the day ordinal is a signed
    integer, so a floor-to-day cast keeps the arithmetic correct."""
    deduper = lk.date_range(format="%Y-%m-%d", window=1)
    deduper.validate("date")

    pairs = list(deduper._gen_similarity_pairs(pa.array(["1969-12-31", "1970-01-01"])))

    assert pairs == [(0, 1)]


def test_date_range_values_reversed_order_still_pairs():
    deduper = lk.date_range(format="%Y-%m-%d", window=2)
    deduper.validate("date")

    pairs = list(deduper._gen_similarity_pairs(pa.array(["2024-01-07", "2024-01-05"])))

    assert sorted(pairs) == [(0, 1)]


def test_sorted_tokens_accepts_large_string_arrays():
    deduper = lk.sorted_tokens()
    deduper.validate("name")

    pairs = list(deduper._gen_similarity_pairs(pa.array(["a b", "b a"], type=pa.large_string())))

    assert pairs == [(0, 1)]


def test_sorted_tokens_null_typed_array_yields_missing_pairs_only():
    """A `null`-typed array (e.g. an all-empty column) carries no tokens:
    only the missing-star pairs emit."""
    deduper = lk.sorted_tokens()
    deduper.validate("name")

    pairs = list(deduper._gen_similarity_pairs(pa.array([None, None], type=pa.null())))

    assert pairs == [(0, 1)]


@pytest.mark.parametrize(
    "values",
    [["2024-01-05", None, None, "2024-01-06"]],
    ids=["none-nulls"],
)
def test_date_range_nulls_pair_with_nulls_only(values):
    """Nulls pair with nulls only, star-shaped; null never pairs with a date.

    The dates sit one day apart and `window=30` would merge them, so the
    only pair is the null-null pair."""
    deduper = lk.date_range(format="%Y-%m-%d", window=30)
    deduper.validate("date")

    pairs = list(deduper._gen_similarity_pairs(pa.array(values)))

    assert pairs == [(1, 2), (0, 3)]


def test_date_range_parse_failure_names_column_and_format():
    """A value that does not parse raises, naming the column and format."""
    deduper = lk.date_range(format="%Y-%m-%d", window=0)
    deduper.validate("birth_date")

    with pytest.raises(InvalidArgumentValueError, match=r"birth_date.*%Y-%m-%d"):
        list(deduper._gen_similarity_pairs(pa.array(["2024-01-05", "05/01/2024"])))


def test_date_range_mixed_formats_fail_under_one_format():
    """Mixed formats are the tar pit the spec forbids solving: the second
    format fails under the first, loudly."""
    deduper = lk.date_range(format="%d/%m/%Y", window=0)
    deduper.validate("date")

    with pytest.raises(InvalidArgumentValueError, match="column 'date'"):
        list(deduper._gen_similarity_pairs(pa.array(["05/01/2024", "2024-01-05"])))


def test_date_range_rejects_non_string_arrays_at_execution():
    deduper = lk.date_range(format="%Y-%m-%d")
    deduper.validate("n")

    with pytest.raises(InvalidArgumentTypeError, match="string"):
        list(deduper._gen_similarity_pairs(pa.array([1, 2])))


def test_date_range_boolean_array_rejected_at_execution():
    deduper = lk.date_range(format="%Y-%m-%d")
    deduper.validate("flag")

    with pytest.raises(InvalidArgumentTypeError, match="string"):
        list(deduper._gen_similarity_pairs(pa.array([True, False])))


def test_date_range_negated_rejects_non_string_array():
    """The negated path type-checks the column too: without the check a
    non-string column fails to parse per value and `~date_range` silently
    matches every row."""
    deduper = ~lk.date_range(format="%Y-%m-%d")
    deduper.validate("n")

    with pytest.raises(InvalidArgumentTypeError, match="string"):
        list(deduper._gen_similarity_pairs(pa.array([1, 2])))


def test_date_range_parse_failure_on_all_values():
    """Every value failing to parse raises, not silently no pairs."""
    deduper = lk.date_range(format="%Y-%m-%d")
    deduper.validate("date")

    with pytest.raises(InvalidArgumentValueError, match="column 'date'"):
        list(deduper._gen_similarity_pairs(pa.array(["nope", "also-nope"])))


def test_date_range_negated_groups_unparseable_values():
    """~date_range groups values that fail to parse (the coherent per-row
    reading: _matches is 'value parses under format')."""
    deduper = ~lk.date_range(format="%Y-%m-%d")
    deduper.validate("date")

    pairs = list(deduper._gen_similarity_pairs(pa.array(["garbage", "also garbage", "2024-01-05"])))

    assert pairs == [(0, 1)]


def test_date_range_negated_never_matches_missing():
    """A missing value matches nothing on either polarity."""
    deduper = ~lk.date_range(format="%Y-%m-%d")
    deduper.validate("date")

    pairs = list(deduper._gen_similarity_pairs(pa.array([None, "2024-01-05"])))

    assert pairs == []


def test_date_range_requires_single_string_column():
    with pytest.raises(InvalidArgumentTypeError):
        lk.date_range(format="%Y-%m-%d").validate(("a", "b"))


def test_date_range_pair_emission_is_deterministic():
    deduper = lk.date_range(format="%Y-%m-%d", window=1)
    deduper.validate("date")

    values = ["2024-01-06", "2024-01-05", "2024-01-06"]

    first = list(deduper._gen_similarity_pairs(pa.array(values)))
    second = list(deduper._gen_similarity_pairs(pa.array(values)))

    assert first == second


def test_date_range_str_renders_format_and_window():
    assert "date_range(format='%Y-%m-%d', window=1)" in str(lk.date_range(format="%Y-%m-%d", window=1))
