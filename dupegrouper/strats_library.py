"""Abstract base class for derived deduplication atrategies

This module contains `BaseStrategy` which provides
`propagate_canonical_id()`, which is at the core functionality of `dupegrouper` and is
used for any deduplication that requires *grouping*. Additionally, the
overrideable `canonicalize()` is defined.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterator
from functools import cache
from typing import TYPE_CHECKING, Any, Protocol, Self, final

import numpy as np
from datasketch import MinHash, MinHashLSH
from networkx.utils.union_find import UnionFind
from numpy.linalg import norm
from rapidfuzz import fuzz
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from typing_extensions import override

from dupegrouper.constants import CANONICAL_ID

if TYPE_CHECKING:
    from dupegrouper.dataframe import LocalDF
    from dupegrouper.types import Columns, Keep, SimilarPairIndices


# BASE STRATEGY:


class BaseStrategyProtocol(Protocol):
    wrapped_df: LocalDF
    keep: Keep

    def set_frame(self, wrapped_df: LocalDF) -> Self: ...
    def set_keep(self, keep: Keep) -> Self: ...
    def _gen_similarity_pairs(self, array: np.ndarray) -> Iterator[SimilarPairIndices]: ...
    def _get_components(self, columns: Columns) -> dict[int, list[int]]: ...
    def canonicalize(self, columns: Columns) -> LocalDF: ...
    def get_array(self, columns: Columns) -> np.ndarray: ...
    def validate(self, columns: Columns) -> None: ...


class BaseStrategy:
    """
    @private

    Defines a deduplication strategy.
    """

    def __init__(self, *args, **kwargs):
        self._init_args = args
        self._init_kwargs = kwargs

    def set_frame(self, wrapped_df: LocalDF) -> Self:
        """Inject dataframe data and load dataframe methods corresponding
        to the type of the dataframe the corresponding methods.

        Args:
            df: The dataframe to set

        Returns:
            self: i.e. allow for further chaining
        """
        self.wrapped_df: LocalDF = wrapped_df
        return self

    def set_keep(self, keep: Keep = "first") -> Self:
        self.keep = keep
        return self

    def _gen_similarity_pairs(self, array: np.ndarray) -> Iterator[SimilarPairIndices]:
        del array  # Unused
        raise NotImplementedError

    
    def _build_union_find(self: BaseStrategyProtocol, columns: Columns) -> dict[int, int]:
        self.validate(columns)
        array = self.get_array(columns)

        n = len(array)

        uf = UnionFind(range(n))
        for i, j in self._gen_similarity_pairs(array):
            uf.union(i, j)

        # print("union find", {i: uf[i] for i in range(n)})

        return uf, n
    
    def _get_components(self, columns: Columns) -> dict[int, list[int]]:
        uf, n = self._build_union_find(columns)

        components = defaultdict(list)
        for i in range(n):
            components[uf[i]].append(i)

        # print("components", components)

        return components

    def canonicalizer(
        self: BaseStrategyProtocol,
        columns: Columns,
        *,
        drop_duplicates: bool,
    ) -> LocalDF:
        canonicals = self.get_array(CANONICAL_ID)
        components: dict[int, list[int]] = self._get_components(columns)

        n = len(canonicals)

        rep_index: dict[int, int] = {}
        for members in components.values():
            if self.keep == "first":
                rep = min(members)
            elif self.keep == "last":
                rep = max(members)

            for i in members:
                rep_index[i] = rep


        new_canonicals = np.array(
            [canonicals[rep_index[i]] for i in range(n)],
            dtype=object,
        )

        self.wrapped_df.put_col(CANONICAL_ID, new_canonicals)

        if not drop_duplicates:
            return self.wrapped_df
        return self.wrapped_df.drop_duplicates(keep=self.keep)


class ColumnArrayMixin:
    """
    @private
    """

    wrapped_df: LocalDF

    def get_array(self, columns: Columns) -> np.ndarray:
        if isinstance(columns, str):
            return np.asarray(self.wrapped_df.get_col(columns), dtype=object)
        elif isinstance(columns, tuple):
            return np.asarray(self.wrapped_df.get_cols(columns), dtype=object)
        else:
            raise TypeError("`columns` must be str or tuple[str]")


class SingleColumnValidationMixin:
    """
    @private
    """

    @staticmethod
    def validate(columns: Any) -> None:
        if not isinstance(columns, str):
            raise ValueError("For single column strategies, `columns` must be defined as a string")


class CompoundColumnValidationMixin:
    """
    @private
    """

    @staticmethod
    def validate(columns: Any) -> None:
        if not isinstance(columns, tuple):
            raise ValueError("For compound columns strategies, `columns` must be defined as a tuple")


# EXACT DEDUPER:


@final
class Exact(BaseStrategy, ColumnArrayMixin):
    """
    @private
    """

    @staticmethod
    def as_is(value):
        return value

    @staticmethod
    def to_tuple(value):
        return tuple(value.tolist())

    @override
    def _get_components(self, columns: Columns) -> dict[int, list[int]]:
        array = self.get_array(columns)

        if isinstance(columns, str):
            key_fn = self.as_is
        else:
            key_fn = self.to_tuple

        components = defaultdict(list)
        for i, v in enumerate(array):
            components[key_fn(v)].append(i)

        return components


# BINARY DEDUPERS:


# TODO: Eventually make `StrMethods` which inherits from `BinaryDedupers`
class BinaryDedupers(BaseStrategy):
    """
    @private
    """

    def __init__(self, pattern: str, case: bool = True):
        super().__init__(pattern=pattern, case=case)
        self._pattern = pattern
        self._case = case

    def _matches(self, value):
        del value  # Unused
        pass

    @override
    def _gen_similarity_pairs(self, array: np.ndarray) -> Iterator[SimilarPairIndices]:
        n = len(array)
        for i in range(n):
            if not self._matches(array[i]):
                continue
            for j in range(i + 1, n):
                if self._matches(array[j]):
                    yield i, j


@final
class StrStartsWith(
    BinaryDedupers,
    ColumnArrayMixin,
    SingleColumnValidationMixin,
):
    """
    @private
    Strings start with canonicalizer.

    Defaults to case sensitive.

    Regex is not supported, please use `StrContains` otherwise.
    """

    @override
    def _matches(self, value: str | None) -> bool:
        if value is None:
            return False
        return (
            value.startswith(self._pattern)
            #
            if self._case
            else value.lower().startswith(self._pattern.lower())
        )


@final
class StrEndsWith(
    BinaryDedupers,
    ColumnArrayMixin,
    SingleColumnValidationMixin,
):
    """
    @private
    Strings start with canonicalizer.

    Defaults to case sensitive.

    Regex is not supported, please use `StrContains` otherwise.
    """

    @override
    def _matches(self, value: str | None) -> bool:
        if value is None:
            return False
        return (
            value.endswith(self._pattern)
            #
            if self._case
            else value.lower().endswith(self._pattern.lower())
        )


@final
class StrContains(
    BinaryDedupers,
    ColumnArrayMixin,
    SingleColumnValidationMixin,
):
    """
    @private
    Strings contains canonicalizer.

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


class ThresholdDedupers(BaseStrategy):
    """
    @private
    """

    def __init__(self, threshold: float = 0.95):
        super().__init__(threshold=threshold)
        self._threshold = threshold

        if not (0 <= threshold < 1):
            raise ValueError("The threshold value must be greater or equal to 0 and less than 1")


@final
class Fuzzy(
    ThresholdDedupers,
    ColumnArrayMixin,
    SingleColumnValidationMixin,
):
    """
    @private
    """

    @staticmethod
    @cache
    def _fuzz_ratio(s1, s2) -> float:
        return fuzz.ratio(s1, s2) / 100

    def _gen_similarity_pairs(self, array: np.ndarray) -> Iterator[SimilarPairIndices]:
        n = len(array)
        for i in range(n):
            for j in range(i + 1, n):
                if self._fuzz_ratio(array[i], array[j]) > self._threshold:
                    yield i, j


@final
class TfIdf(
    ThresholdDedupers,
    ColumnArrayMixin,
    SingleColumnValidationMixin,
):
    """
    @private
    TF-IDF canonicalizer.

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
        ngram_range = (self._ngram, self._ngram) if isinstance(self._ngram, int) else self._ngram

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

    def _gen_similarity_pairs(self, array: np.ndarray) -> Iterator[SimilarPairIndices]:
        """Extract arrays based on similarity scores

        Filter's out _approximate_ perfect scores (i.e. decimal handling) and
        loads up results into a tuple of arrays"""
        sparse = self._get_sparse_matrix(array)

        sparse_coo = sparse.tocoo()

        rows, cols = sparse_coo.row, sparse_coo.col

        for i in range(len(rows)):
            yield rows[i], cols[i]


@final
class LSH(
    ThresholdDedupers,
    ColumnArrayMixin,
    SingleColumnValidationMixin,
):
    """
    @private
    """

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

    def _gen_token(self, text) -> Iterator:
        for i in range(len(text) - self._ngram + 1):
            yield text[i : i + self._ngram]

    def _build_minhashes(self, array: np.ndarray) -> list[MinHash]:
        minhashes: list[MinHash] = []
        for value in array:
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

    def _gen_similarity_pairs(self, array: np.ndarray) -> Iterator[SimilarPairIndices]:

        minhashes: list[MinHash] = self._build_minhashes(array)
        lsh: MinHashLSH = self._lsh(minhashes)

        for idx, minhash in enumerate(minhashes):
            for idy in lsh.query(minhash):
                if idx < idy:
                    yield idx, idy


# COMPOUND COLUMN:


@final
class Jaccard(
    ThresholdDedupers,
    ColumnArrayMixin,
    CompoundColumnValidationMixin,
):
    """
    @private
    """

    def _gen_similarity_pairs(self, array: np.ndarray) -> Iterator[SimilarPairIndices]:
        sets = [set(row) for row in array]

        n = len(array)
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


@final
class Cosine(
    ThresholdDedupers,
    ColumnArrayMixin,
    CompoundColumnValidationMixin,
):
    """
    @private
    """

    def _gen_similarity_pairs(self, array: np.ndarray) -> Iterator[SimilarPairIndices]:
        n = len(array)
        for idx in range(n):
            for idy in range(idx + 1, n):
                product = np.dot(array[idx], array[idy])

                if not product:
                    continue  # no match

                norms = norm(array[idx]) * norm(array[idy])

                if not norms:
                    continue  # zero div: guardrail

                if product / norms > self._threshold:
                    yield idx, idy


# PUBLIC:


def exact() -> BaseStrategy:
    """TODO"""
    return Exact()


def str_startswith(pattern: str, case: bool = True) -> BaseStrategy:
    """TODO"""
    return StrStartsWith(pattern=pattern, case=case)


def str_endswith(pattern: str, case: bool = True) -> BaseStrategy:
    """TODO"""
    return StrEndsWith(pattern=pattern, case=case)


def str_contains(
    pattern: str,
    case: bool = True,
    regex: bool = False,
) -> BaseStrategy:
    """TODO"""
    return StrContains(pattern=pattern, case=case, regex=regex)


def fuzzy(threshold: float = 0.95) -> BaseStrategy:
    """TODO"""
    return Fuzzy(threshold=threshold)


def tfidf(
    threshold: float = 0.95,
    ngram: int | tuple[int, int] = 3,
    topn: int = 2,
) -> BaseStrategy:
    """TODO"""
    return TfIdf(threshold=threshold, ngram=ngram, topn=topn)


def lsh(
    threshold: float = 0.95,
    ngram: int = 3,
    num_perm: int = 128,
) -> BaseStrategy:
    """TODO"""
    return LSH(threshold=threshold, ngram=ngram, num_perm=num_perm)


def jaccard(threshold: float = 0.95) -> BaseStrategy:
    """TODO"""
    return Jaccard(threshold=threshold)


def cosine(threshold: float = 0.95) -> BaseStrategy:
    """TODO"""
    return Cosine(threshold=threshold)
