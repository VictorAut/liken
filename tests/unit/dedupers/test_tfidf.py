"""Liken deduper unit tests"""

import math

import pyarrow as pa
import pytest

import liken as lk
from liken.exceptions import InvalidArgumentTypeError
from liken.exceptions import InvalidArgumentValueError


@pytest.mark.parametrize("ngram", [1, 2, 3])
def test_tfidf_null_never_pairs_with_value(ngram):
    """A null pairs only with another null, at any ngram.

    "nathan" and "nadia" share no character 3-grams, so at the default ngram
    the null-null pair is the only pair; the assertion allows value pairs but
    forbids any pair touching a null other than the null-null pair.
    """
    array = pa.array(["nathan", None, "nadia", None])

    pairs = sorted(lk.tfidf(ngram=ngram)._gen_similarity_pairs(array))

    assert (1, 3) in pairs
    assert all(1 not in pair and 3 not in pair for pair in pairs if pair != (1, 3))


def test_tfidf_all_null_input_yields_star_pairs():
    """All-null input yields star-shaped null pairs without touching the vectoriser."""
    array = pa.array([None, None, None])

    assert list(lk.tfidf()._gen_similarity_pairs(array)) == [(0, 1), (0, 2)]


###################################
# tfidf construction validation  #
##################################


@pytest.mark.parametrize("topn", [0, -1])
def test_tfidf_rejects_non_positive_topn(topn):
    """`topn` below 1 is refused at construction: an unvalidated topn reaches
    the native library and can abort the process."""
    with pytest.raises(InvalidArgumentValueError, match="`topn` must be an integer in the range \\[1, inf\\), got"):
        lk.tfidf(topn=topn)


@pytest.mark.parametrize("topn", [2.0, "2", True, None])
def test_tfidf_rejects_non_integer_topn(topn):
    with pytest.raises(InvalidArgumentTypeError, match="`topn` must be an integer, got"):
        lk.tfidf(topn=topn)


@pytest.mark.parametrize("ngram", [0, -1])
def test_tfidf_rejects_non_positive_ngram(ngram):
    with pytest.raises(InvalidArgumentValueError, match="`ngram` must be an integer in the range \\[1, inf\\), got"):
        lk.tfidf(ngram=ngram)


@pytest.mark.parametrize("ngram", ["3", 3.0, True, None])
def test_tfidf_rejects_non_integer_ngram(ngram):
    with pytest.raises(InvalidArgumentTypeError, match="`ngram` must be an integer or a 2-tuple of integers, got"):
        lk.tfidf(ngram=ngram)


@pytest.mark.parametrize("ngram", [(2, 1), (0, 2), (1, 0), (2, 2.0), (2, "3")])
def test_tfidf_rejects_invalid_ngram_tuple(ngram):
    """A 2-tuple ngram range must be integers with 1 <= low <= high."""
    with pytest.raises((InvalidArgumentTypeError, InvalidArgumentValueError)):
        lk.tfidf(ngram=ngram)


@pytest.mark.parametrize("ngram", [(2,), (1, 2, 3), []])
def test_tfidf_rejects_wrong_shape_ngram_tuple(ngram):
    with pytest.raises(InvalidArgumentTypeError, match="`ngram` must be an integer or a 2-tuple of integers, got"):
        lk.tfidf(ngram=ngram)


def test_tfidf_accepts_valid_ngram_and_topn():
    lk.tfidf(ngram=3, topn=2)
    lk.tfidf(ngram=(1, 2), topn=1)


def test_tfidf_matches_a_pair_exactly_at_the_threshold():
    """A pair whose similarity equals the threshold matches.

    The threshold is lowered by one ULP before it reaches the delegate,
    whose comparison is strict, so a pair stored at exactly the threshold
    survives.
    """
    values = ["a b", "a c", "a b c"]

    probe = lk.tfidf(ngram=1, topn=5, threshold=math.nextafter(0.0, 1.0))
    sparse = probe._get_sparse_matrix(values).tocoo()
    row, col, value = max(
        ((int(r), int(c), float(v)) for r, c, v in zip(sparse.row, sparse.col, sparse.data, strict=False) if r < c),
        key=lambda item: item[2],
    )

    pairs = list(lk.tfidf(ngram=1, topn=5, threshold=value)._gen_similarity_pairs(pa.array(values)))

    assert (row, col) in pairs

    above = math.nextafter(value, math.inf)
    pairs = list(lk.tfidf(ngram=1, topn=5, threshold=above)._gen_similarity_pairs(pa.array(values)))

    assert (row, col) not in pairs
