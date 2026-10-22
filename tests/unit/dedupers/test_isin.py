"""Liken deduper unit tests"""

from unittest.mock import Mock

import pyarrow as pa
import pytest

import liken as lk
from liken.dedupers.isin import isin
from liken.dedupers.str_startswith import StrStartsWith
from liken.exceptions import InvalidArgumentTypeError


@pytest.mark.parametrize("values", [["a"], ["a", None]], ids=["null-unlisted", "null-listed"])
def test_negated_isin_never_matches_null(mock_df, values):
    """A missing value never satisfies a negated predicate.

    The null stays out of every group, whether or not None is in `values`:
    the negation must not consult the inner membership test for a missing
    value.
    """
    mock_df.get_array = Mock(return_value=pa.array([None, "a", None, "b"]))

    uf, n = (~isin(values)).set_frame(mock_df).build_union_find("address", [])

    assert n == 4
    assert uf[0] != uf[1] and uf[0] != uf[3]  # null vs each value
    assert uf[0] != uf[2]  # nulls do not group under a negated predicate
    assert uf[2] != uf[1] and uf[2] != uf[3]


def test_negated_isin_never_matches_nan(mock_df):
    """NaN is missing and never satisfies a negated predicate."""
    mock_df.get_array = Mock(return_value=pa.array([float("nan"), 1.0, 2.0]))

    uf, n = (~isin([1.0])).set_frame(mock_df).build_union_find("address", [])

    assert n == 3
    assert uf[0] != uf[1] and uf[0] != uf[2]


def test_negated_str_startswith_vectorized_skips_nulls():
    """The vectorised negated path leaves missing values unmatched."""
    array = pa.array(["alpha", None, "beta", "gamma"])

    pairs = list((~StrStartsWith(pattern="al"))._gen_similarity_pairs(array))

    assert pairs == [(2, 3)]  # "beta" and "gamma" only; the null is skipped


@pytest.mark.parametrize(
    "values, null_grouped",
    [(["a"], False), (["a", None], True)],
    ids=["null-unlisted", "null-listed"],
)
def test_isin_null_matches_iff_none_is_listed(mock_df, values, null_grouped):
    """Positive isin keeps Python membership: a null matches iff None is in values."""
    mock_df.get_array = Mock(return_value=pa.array([None, "a", None, "b"]))

    uf, n = isin(values).set_frame(mock_df).build_union_find("address", [])

    assert n == 4
    if null_grouped:
        assert uf[0] == uf[1] == uf[2]  # None listed: the nulls match
        assert uf[0] != uf[3]
    else:
        assert uf[0] != uf[1] and uf[0] != uf[2] and uf[1] != uf[2]  # only "a" matches


###############################
# isin construction validation #
###############################


@pytest.mark.parametrize(
    "values",
    ["london", 42, {"london": 1}, object()],
    ids=["str", "int", "dict", "object"],
)
def test_isin_rejects_non_collection_values(values):
    """`values` must be a list, tuple, set or frozenset; a str is an iterable
    and was silently treated as a character collection."""
    with pytest.raises(
        InvalidArgumentTypeError,
        match="`values` must be of type 'list' or 'tuple' or 'set' or 'frozenset', got",
    ):
        lk.isin(values=values)


@pytest.mark.parametrize(
    "values",
    [["london"], ("london",), {"london"}, frozenset({"london"})],
    ids=["list", "tuple", "set", "frozenset"],
)
def test_isin_materialises_values_to_tuple(values):
    """Values are materialised at construction, so a lazy iterable cannot be
    consumed by the first match."""
    deduper = lk.isin(values=values)

    assert deduper._values == ("london",)
