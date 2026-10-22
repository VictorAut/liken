"""Liken deduper unit tests"""

import pyarrow as pa
import pytest

import liken as lk
from liken.exceptions import InvalidArgumentTypeError
from liken.exceptions import InvalidArgumentValueError


@pytest.mark.parametrize("ngram", [1, 2, 3])
def test_lsh_null_never_pairs_with_value(ngram):
    """A null pairs only with another null, at any ngram."""
    array = pa.array(["nathan", None, "nadia", None])

    pairs = sorted(lk.lsh(ngram=ngram)._gen_similarity_pairs(array))

    assert (1, 3) in pairs
    assert all(1 not in pair and 3 not in pair for pair in pairs if pair != (1, 3))


def test_lsh_all_null_input_yields_star_pairs():
    """All-null input yields star-shaped null pairs without touching datasketch."""
    array = pa.array([None, None, None])

    assert list(lk.lsh()._gen_similarity_pairs(array)) == [(0, 1), (0, 2)]


###############################
# lsh construction validation #
###############################


@pytest.mark.parametrize("ngram", [0, -1])
def test_lsh_rejects_non_positive_ngram(ngram):
    """`ngram` below 1 silently collapses the column at execution; refused at construction."""
    with pytest.raises(InvalidArgumentValueError, match="`ngram` must be an integer in the range \\[1, inf\\), got"):
        lk.lsh(ngram=ngram)


@pytest.mark.parametrize("ngram", ["3", 3.0, True, None])
def test_lsh_rejects_non_integer_ngram(ngram):
    with pytest.raises(InvalidArgumentTypeError, match="`ngram` must be an integer, got"):
        lk.lsh(ngram=ngram)


@pytest.mark.parametrize("num_perm", [0, -5])
def test_lsh_rejects_non_positive_num_perm(num_perm):
    with pytest.raises(InvalidArgumentValueError, match="`num_perm` must be an integer in the range \\[1, inf\\), got"):
        lk.lsh(num_perm=num_perm)


@pytest.mark.parametrize("num_perm", [128.0, "128", True, None])
def test_lsh_rejects_non_integer_num_perm(num_perm):
    with pytest.raises(InvalidArgumentTypeError, match="`num_perm` must be an integer, got"):
        lk.lsh(num_perm=num_perm)


def test_lsh_accepts_valid_ngram_and_num_perm():
    lk.lsh(ngram=3, num_perm=128)
