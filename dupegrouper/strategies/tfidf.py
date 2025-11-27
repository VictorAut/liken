"""Perform near deduplication with TF-IDF"""

from __future__ import annotations
import functools
import logging
from typing_extensions import override
import typing

import numpy as np

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sparse_dot_topn import sp_matmul_topn  # type: ignore

from dupegrouper.definitions import TMP_ATTR_LABEL, SeriesLike
from dupegrouper.wrappers import WrappedDataFrame
from dupegrouper.strategy import DeduplicationStrategy


# LOGGER:


logger = logging.getLogger(__name__)


# TFIDF:


class TfIdf(DeduplicationStrategy):
    """TF-IDF deduper.

    Note: high "top N" numbers at initialisation may cause spurious results.
    Use with care!

    Note: Whilst powerful, this is computationally intensive: ~*O(n^2)*

    Additional keywords arguments can be passed to parametrise the vectorizer,
    as listed in the [TF-IDF vectorizer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
    """

    def __init__(
        self,
        ngram: int | tuple[int, int] = 3,
        tolerance: float = 0.05,
        topn: int = 2,
        **kwargs,
    ):
        super().__init__(
            ngram=ngram,
            tolerance=tolerance,
            topn=topn,
            **kwargs,
        )
        self._ngram = ngram
        self._tolerance = tolerance
        self._topn = topn
        self._kwargs = kwargs

    @functools.singledispatchmethod
    def _vectorize(self, ngram) -> TfidfVectorizer:
        """Parametriser tf-idf vectorizer

        Dispatches the n-gram ranging allowing for passing as an integer i.e.
        3 -> (3, 3) or over via a tuple i.e. (2, 5).

        Args:
            ngram: the n-gram range"""
        del ngram  # Unused by generic
        raise TypeError("ngram must be of type int or a length 2 tuple of integers")

    @_vectorize.register(int)
    def _(self, ngram) -> TfidfVectorizer:
        return TfidfVectorizer(
            analyzer="char",
            ngram_range=(ngram, ngram),
            **self._kwargs,
        )

    @_vectorize.register(tuple)
    def _(self, ngram) -> TfidfVectorizer:
        return TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram,
            **self._kwargs,
        )

    def _get_similarities_matrix(
        self,
        vectorizer: TfidfVectorizer,
        array: np.ndarray,
        /,
    ) -> csr_matrix:
        """sparse matrix of similarities, given the "top N" _best_ matches"""
        return sp_matmul_topn(
            (mat := vectorizer.fit_transform(array)),
            mat.T,
            top_n=self._topn,
            threshold=1 - self._tolerance,
            sort=True,
        )

    @staticmethod
    def _get_matches_array(
        sparse: csr_matrix,
        array: np.ndarray,
        /,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract arrays based on similarity scores

        Filter's out _approximate_ perfect scores (i.e. decimal handling) and
        loads up results into a tuple of arrays"""
        sparse_coo = sparse.tocoo()

        # floating point precision handling of perfect match filter
        mask = ~np.isclose(sparse_coo.data, 1.0)

        rows, cols = sparse_coo.row[mask], sparse_coo.col[mask]

        n = len(rows)

        lookup = np.empty(n, dtype=object)
        match = np.empty(n, dtype=object)
        similarity = np.zeros(n)

        for i in range(0, n):

            lookup[i] = array[rows[i]]
            match[i] = array[cols[i]]
            similarity[i] = sparse[rows[i], cols[i]]

        return lookup, match, similarity

    @staticmethod
    def _gen_map(
        matches: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> typing.Iterator[dict[str, str]]:
        """Generate unique matches as a map

        Given a "top N" > 1 get similarity matches in ascending order, thus
        guaranteeing that the _best_ match will be the last match, as retrived
        by the public API.

        Args:
            matches: a tuple array of "original" values, as well as a matched
            value for the equivalent index of that array.

        Yields:
            A map, e.g. {"example_address_1": "example_address_2"}, will be
            yielded IF it has not been yielded before *or*, the prior seen
            instance of the map match is not the "inverse" map.

            i.e. the mapping
                {"example_address_1": "example_address_2"}
            will *not* be yielded if
                {"example_address_2": "example_address_2"}
            has already been yielded.
        """
        seen: set[tuple[int, int]] = set()

        # iter: reorder to similarities score ascending
        for i, j, _ in zip(*map(np.flip, matches)):

            # not inversed; not repeated
            if {(i, j), (j, i)}.isdisjoint(seen):
                seen.add((i, j))
                yield {i: j}

    @override
    def dedupe(self, attr: str, /) -> WrappedDataFrame:
        """Deduplicate with term frequency, inverse document frequency.

        Here, 1-to-1 maps are identified using the procedure i.e. for *each*
        row, *not* a map across the whole array as identified by `attr`. This
        is because this deduper implements "top N" matches functionality and a
        non-iterative approach would produce a mapping object that > len(array)

        Therefore, for instances of "top N" matches where > 1, 1-to-1 match
        maps are iteratively applied from worst to best, guarnateeing the best
        end match.
        """
        logger.debug(
            f'Deduping attribute "{attr}" with {self.__class__.__name__}('
            f"ngram={self._ngram}, "
            f"tolerance={self._tolerance}, "
            f"topn={self._topn}"
            ")"
        )

        vectorizer = self._vectorize(self._ngram)

        similarities = self._get_similarities_matrix(vectorizer, np.array(self.wrapped_df.get_col(attr)))

        matches = self._get_matches_array(similarities, np.array(self.wrapped_df.get_col(attr)))

        for tfidf_map in self._gen_map(matches):

            attr_map: SeriesLike = self.wrapped_df.map_dict(attr, tfidf_map)

            new_attr: SeriesLike = self.wrapped_df.fill_na(attr_map, self.wrapped_df.get_col(attr))

            self.wrapped_df.put_col(TMP_ATTR_LABEL, new_attr)

            self.assign_group_id(TMP_ATTR_LABEL).drop_col(TMP_ATTR_LABEL)

        return self.wrapped_df
