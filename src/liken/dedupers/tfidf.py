"""tfidf deduper"""

from collections.abc import Iterator
from typing import Any
from typing import ClassVar
from typing import final

import pyarrow as pa
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from liken.core.deduper import BaseDeduper
from liken.core.deduper import SingleColumnMixin
from liken.core.deduper import ThresholdDeduper
from liken.core.registries import dedupers_registry
from liken.types import SimilarPairIndices


@final
class TfIdf(
    SingleColumnMixin,
    ThresholdDeduper,
):
    """
    TF-IDF deduper.

    Additional keywords arguments can be passed to parametrise the vectorizer,
    as listed in the [TF-IDF vectorizer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
    """

    _NAME: ClassVar[str] = "tfidf"

    def __init__(
        self,
        ngram: int | tuple[int, int] = 3,
        threshold: float = 0.95,
        topn: int = 2,
        **kwargs: Any,
    ):
        super().__init__(
            threshold=threshold,
            ngram=ngram,
            topn=topn,
            **kwargs,
        )
        self._ngram = ngram
        self._threshold = threshold
        self._topn = topn
        self._kwargs = kwargs

    def _vectorize(self) -> TfidfVectorizer:
        ngram_range = (self._ngram, self._ngram) if isinstance(self._ngram, int) else self._ngram

        return TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            **self._kwargs,
        )

    def _get_sparse_matrix(self, array: list) -> csr_matrix:
        """sparse matrix of similarities, given the top N best matches"""

        vectorizer = self._vectorize()
        matrix = vectorizer.fit_transform(array)
        return sp_matmul_topn(
            matrix,
            matrix.T,
            top_n=self._topn,
            threshold=self._threshold,
            sort=True,
        )

    def _gen_similarity_pairs(self, array: pa.Array) -> Iterator[SimilarPairIndices]:
        """Extract arrays based on similarity scores

        Filter's out _approximate_ perfect scores (i.e. decimal handling) and
        loads up results into a tuple of arrays"""
        array: list = array.to_pylist()

        sparse = self._get_sparse_matrix(array)

        sparse_coo = sparse.tocoo()

        rows, cols = sparse_coo.row, sparse_coo.col

        for i in range(len(rows)):
            yield rows[i], cols[i]

    def __str__(self):
        return self.str_representation(self._NAME)


@dedupers_registry.register("tfidf")
def tfidf(
    threshold: float = 0.95,
    ngram: int | tuple[int, int] = 3,
    topn: int = 2,
    **kwargs: Any,
) -> BaseDeduper:
    """Near string deduplication using term frequency, inverse document
    frequency.

    Usage is on single columns of a dataframe. `tfidf` is a tuneable deduper.
    Experimentation is required for optimal use.

    Args:
        threshold: the minimum threshold at which similarity between two pairs
            of values will be considered valid for deduplication.
        ngram: the number of character ngrams to consider. For the `tfidf`
            implementation this is the ngram bounded range. If you pass this as
            an integer you are saying the bounds are the same. E.g. `ngram=1`
            is equivalent to the range bounded over (1, 1) (i.e. unigrams
            only). `ngram=(1, 2)` is unigrams and bigrams. Increasing ngrams
            reduces overall deduplication. However, too small an `ngram` may
            result in false positives.
        topn: the number of best matches to consider when building similarity
            matrices.
        **kwargs: additional kwargs as accepted in sklearn's [Tfidf
            Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

    Returns:
        Instance of `BaseDeduper`.

    Example:
        Applied to a single column:

            import liken as lk

            df = (
                lk.dedupe(df)
                .apply({"address": tfidf(threshold=0.8, ngram=1)})
                .drop_duplicates(keep="last")
            )

        E.g.

            >>> df # Before
            +------+-----------+----------------------+
            | id   |  address  |         email        |
            +------+-----------+----------------------+
            |  1   |  london   |  fizzpop@gmail.com   |
            |  2   |   null    |  foobar@gmail.com    |
            |  3   |  london   |  foobar@gmail.co.uk  |
            +------+-----------+----------------------+

            >>> df # After
            +------+-----------+----------------------+
            | id   |  address  |         email        |
            +------+-----------+----------------------+
            |  1   |  london   |  fizzpop@gmail.com   |
            |  3   |  london   |  foobar@gmail.co.uk  |
            +------+-----------+----------------------+

        Note that the same deduper with `ngram=2` does not deduplicate any
        records in the above example.
    """
    return TfIdf(threshold=threshold, ngram=ngram, topn=topn, **kwargs)
