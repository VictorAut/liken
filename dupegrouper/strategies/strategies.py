import functools
import hashlib
import logging
import re
import typing
from typing_extensions import override

from datasketch import MinHash, MinHashLSH
import numpy as np
from numpy.linalg import norm
from rapidfuzz import fuzz
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sparse_dot_topn import sp_matmul_topn  # type: ignore

from dupegrouper.definitions import (
    HASH_ATTR_LABEL,
    TMP_ATTR_LABEL,
    SeriesLike,
)
from dupegrouper.wrappers import WrappedDataFrame
from dupegrouper.strategy import DeduplicationStrategy


# LOGGER:


logger = logging.getLogger(__name__)


# TYPES:


_T = typing.TypeVar("_T")


# EXACT DEDUPER:


class Exact(DeduplicationStrategy):

    @override
    def dedupe(self, columns: str, /) -> WrappedDataFrame:
        return self.canonicalize(columns)


# BINARY DEDUPERS:


class BinaryDedupers(DeduplicationStrategy):
    """TODO"""

    def __init__(self, pattern: str, case: bool = True):
        self._pattern = pattern
        self._case = case

    def _matches(self, value):
        del value  # Unused
        pass

    @staticmethod
    def get_matches(
        match_fn: typing.Callable[[str], bool],
        attr: np.ndarray,
    ) -> dict[str, str]:
        match_map = {}
        for key in attr:
            for value in attr:
                if match_fn(key) and match_fn(value):
                    match_map[key] = value
                    break
        return match_map

    @override
    def dedupe(self, columns: str, /) -> WrappedDataFrame:

        attr_array: np.ndarray = np.unique(self.wrapped_df.get_col(columns))
        match_map: dict[str, str] = self.get_matches(self._matches, attr_array)
        new_attr: SeriesLike = self.wrapped_df.map_dict(columns, match_map)
        self.wrapped_df.put_col(TMP_ATTR_LABEL, new_attr)
        self.canonicalize(TMP_ATTR_LABEL, include_exact=False)
        self.wrapped_df.drop_col(TMP_ATTR_LABEL)

        return self.wrapped_df


class StrStartsWith(BinaryDedupers):
    """Strings start with deduper.

    Defaults to case sensitive.

    Regex is not supported, please use `StrContains` otherwise.
    """

    @override
    def _matches(self, value: str) -> bool:
        return (
            value.startswith(self._pattern)
            #
            if self._case
            else value.lower().startswith(self._pattern.lower())
        )


class StrEndsWith(BinaryDedupers):
    """Strings start with deduper.

    Defaults to case sensitive.

    Regex is not supported, please use `StrContains` otherwise.
    """

    @override
    def _matches(self, value: str) -> bool:
        return (
            value.endswith(self._pattern)
            #
            if self._case
            else value.lower().endswith(self._pattern.lower())
        )


class StrContains(BinaryDedupers):
    """Strings contains deduper.

    Defaults to case sensitive. Supports literal substring or regex search.
    """

    def __init__(self, pattern: str, case: bool = True, regex: bool = False):
        super().__init__(pattern=pattern, case=case)
        self._regex = regex

        if self._regex:
            flags = 0 if self._case else re.IGNORECASE
            self._compiled_pattern = re.compile(self._pattern, flags)

    @override
    def _matches(self, value: str) -> bool:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return False

        if self._regex:
            return bool(self._compiled_pattern.search(value))
        else:
            if self._case:
                return self._pattern in value
            else:
                return self._pattern.lower() in value.lower()


# THRESHOLD DEDUPERS:


class ThresholdDedupers(DeduplicationStrategy):
    def __init__(self, threshold: float = 0.95):
        super().__init__(threshold=threshold)
        self._threshold = threshold

    def _gen_similarity_indices(self, attr) -> typing.Iterator[tuple[int, int]]:
        del attr  # Unused
        pass


# SINGLE COLUMN:


class SingleColumn(ThresholdDedupers):
    @override
    def dedupe(self, columns: str, /) -> WrappedDataFrame:

        attr: np.ndarray = np.asarray(self.wrapped_df.get_col(columns))

        for idx, idy in self._gen_similarity_indices(attr):
            indice_map: dict[str, str] = {attr[idx]: attr[idy]}
            new_attr: SeriesLike = self.wrapped_df.map_dict(columns, indice_map)
            new_attr_filled: SeriesLike = self.wrapped_df.fill_na(new_attr, self.wrapped_df.get_col(columns))
            self.wrapped_df.put_col(TMP_ATTR_LABEL, new_attr_filled)
            self.canonicalize(TMP_ATTR_LABEL)
            self.wrapped_df.drop_col(TMP_ATTR_LABEL)

        return self.wrapped_df


class Fuzzy(SingleColumn):

    @staticmethod
    @functools.cache
    def _fuzz_ratio(s1, s2) -> float:
        return fuzz.ratio(s1, s2) / 100

    def _gen_similarity_indices(self, attr) -> typing.Iterator[tuple[int, int]]:
        n = len(attr)
        for i in range(n):
            for j in range(i + 1, n):
                if self._fuzz_ratio(attr[i], attr[j]) > self._threshold:
                    yield i, j


class TfIdf(SingleColumn):
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
        threshold: float = 0.95,
        topn: int = 2,
        **kwargs,
    ):
        super().__init__(
            threshold=threshold,
            **kwargs,
        )
        self._ngram = ngram
        self._threshold = threshold
        self._topn = topn
        self._kwargs = kwargs

    def _vectorize(self) -> TfidfVectorizer:
        
        if isinstance(self._ngram, int):
            ngram_range = (self._ngram, self._ngram)
        else:
            ngram_range = self._ngram

        return TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            **self._kwargs,
        )

    def _get_sparse_matrix(self, array: np.ndarray) -> csr_matrix:
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

    def _gen_similarity_indices(self, attr) -> typing.Iterator[tuple[int, int]]:
        """Extract arrays based on similarity scores

        Filter's out _approximate_ perfect scores (i.e. decimal handling) and
        loads up results into a tuple of arrays"""
        sparse = self._get_sparse_matrix(attr)

        sparse_coo = sparse.tocoo()

        # Handle floating point precision errors
        mask = ~np.isclose(sparse_coo.data, 1.0)

        rows, cols = sparse_coo.row[mask], sparse_coo.col[mask]

        for i in range(len(rows)):
            yield rows[i], cols[i]


class Lsh(SingleColumn):
    """TODO"""

    def __init__(
        self,
        ngram: int = 3,
        num_perm: int = 128,
        threshold: float = 0.95,
    ):
        super().__init__(
            threshold=threshold,
        )
        self._ngram = ngram
        self._threshold = threshold
        self._num_perm = num_perm

    def _gen_token(self, text) -> typing.Iterator:
        for i in range(len(text) - self._ngram + 1):
            yield text[i : i + self._ngram]

    def _build_minhashes(self, attr) -> list[MinHash]:
        minhashes: list[MinHash] = []
        for value in attr:
            m = MinHash(num_perm=self._num_perm)
            for token in self._gen_token(value):
                m.update(token.encode("utf8"))
            minhashes.append(m)
        return minhashes

    def _lsh(self, minhashes: list[MinHash]) -> MinHashLSH:
        lsh = MinHashLSH(
            threshold=self._threshold,
            num_perm=self._num_perm,
        )

        for i, m in enumerate(minhashes):
            lsh.insert(i, m)

        return lsh

    def _gen_similarity_indices(self, attr) -> typing.Iterator[tuple[int, int]]:

        minhashes: list[MinHash] = self._build_minhashes(attr)
        lsh: MinHashLSH = self._lsh(minhashes)

        for idx, minhash in enumerate(minhashes):
            for idy in lsh.query(minhash):
                if idx < idy:
                    yield idx, idy


# COMPOUND COLUMN:


class CompoundColumn(ThresholdDedupers):

    @staticmethod
    def _hash(value: typing.Any) -> str:
        """deterministic hash for reproducability; order sensitive"""
        return hashlib.sha256(value.tobytes()).hexdigest()

    @override
    def dedupe(self, columns: typing.Iterable[str], /) -> WrappedDataFrame:
        """TODO"""

        attrs = np.asarray(self.wrapped_df.get_cols(columns))

        hash_attrs: SeriesLike = [self._hash(i) for i in attrs]

        self.wrapped_df.put_col(HASH_ATTR_LABEL, hash_attrs)

        for idx, idy in self._gen_similarity_indices(attrs):
            indice_map: dict[str, str] = {hash_attrs[idx]: hash_attrs[idy]}
            new_attr: SeriesLike = self.wrapped_df.map_dict(HASH_ATTR_LABEL, indice_map)
            new_attr_filled: SeriesLike = self.wrapped_df.fill_na(new_attr, self.wrapped_df.get_col(HASH_ATTR_LABEL))
            self.wrapped_df.put_col(TMP_ATTR_LABEL, new_attr_filled)
            self.canonicalize(TMP_ATTR_LABEL)
            self.wrapped_df.drop_col(TMP_ATTR_LABEL)

        self.wrapped_df.drop_col(HASH_ATTR_LABEL)
        return self.wrapped_df


class Jaccard(CompoundColumn):

    def _gen_similarity_indices(self, attrs: np.ndarray) -> typing.Iterator[tuple[int, int]]:
        sets = [set(row) for row in attrs]

        n = len(attrs)
        for idx in range(n):
            for idy in range(idx + 1, n):
                intersection = sets[idx] & sets[idy]

                if not intersection:
                    continue  # no match

                union = sets[idx] | sets[idy]

                if not union:
                    continue  # zero div: guardrail

                if len(intersection) / len(union) > self._threshold:
                    yield idx, idy


class Cosine(CompoundColumn):

    def _gen_similarity_indices(self, attrs: np.ndarray) -> typing.Iterator[tuple[int, int]]:
        n = len(attrs)
        for idx in range(n):
            for idy in range(idx + 1, n):
                product = np.dot(attrs[idx], attrs[idy])

                if not product:
                    continue  # no match

                norms = norm(attrs[idx]) * norm(attrs[idy])

                if not norms:
                    continue  # zero div: guardrail

                if product / norms > self._threshold:
                    yield idx, idy


# CUSTOM:


class Custom(DeduplicationStrategy):

    def __init__(
        self,
        func: typing.Callable[..., dict[_T, _T]],
        attr: str,
        /,
        **kwargs,
    ):
        super().__init__(
            func,
            attr,
            **kwargs,
        )
        self._func = func
        self._attr = attr
        self._kwargs = kwargs

    @override
    def dedupe(self, attr=None) -> WrappedDataFrame:
        """dedupe with custom defined callable

        Implements deduplication using a function defined _outside_ of the
        scope of this library i.e. by the end user.

        The function signature must be of the following style:

        `my_func(df, attr, /, **kwargs)`

        Where `df` is the dataframe, `attr` is a string identifying the label
        of the dataframe attribute requiring deduplication and kwargs are any
        number of additional keyword arguments taken by the function

        `df` and `attr`, must be *positional* arguments in the correct order!
        """
        del attr  # Unused
        logger.debug(
            f'Deduping attribute "{self._attr}" with {self._func.__name__}'
            f'({", ".join(f"{k}={v}" for k, v in self._kwargs.items())})'
        )

        new_attr: SeriesLike = self.wrapped_df.map_dict(
            self._attr,
            self._func(
                self.wrapped_df,
                self._attr,
                **self._kwargs,
            ),
        )

        self.wrapped_df.put_col(TMP_ATTR_LABEL, new_attr)

        return self.canonicalize(TMP_ATTR_LABEL).drop_col(TMP_ATTR_LABEL)