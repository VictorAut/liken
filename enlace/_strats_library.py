"""Abstract base class for derived deduplication atrategies

This module contains `BaseStrategy` which provides
`propagate_canonical_id()`, which is at the core functionality of `enlace` and is
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
import pandas as pd
from datasketch import MinHash, MinHashLSH
from networkx.utils.union_find import UnionFind
from numpy.linalg import norm
from rapidfuzz import fuzz
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from typing_extensions import override

from enlace._constants import CANONICAL_ID


if TYPE_CHECKING:
    from enlace._dataframe import LocalDF
    from enlace._types import UF, Columns, Keep, SimilarPairIndices
    from enlace._executors import SingleComponents, MultiComponents


# BASE STRATEGY:


class BaseStrategyProtocol(Protocol):
    wdf: LocalDF
    with_na_placeholder: bool

    def set_frame(self, wdf: LocalDF) -> Self: ...
    def _gen_similarity_pairs(self, array: np.ndarray) -> Iterator[SimilarPairIndices]: ...
    def build_union_find(self, columns: Columns) -> tuple[UF, int]: ...
    def canonicalizer(
        self,
        *,
        components: SingleComponents | MultiComponents,
        drop_duplicates: bool,
        keep: Keep,
    ) -> LocalDF: ...
    def str_representation(self, name: str) -> str: ...
    
    # available with mixin
    def validate(self, columns: Columns) -> None: ...


class BaseStrategy(BaseStrategyProtocol):
    """
    @private

    Defines a deduplication strategy.
    """

    with_na_placeholder: bool = True  # TODO document this

    def __init__(self, *args, **kwargs):
        self._init_args = args
        self._init_kwargs = kwargs

    def set_frame(self, wdf: LocalDF) -> Self:
        """Inject dataframe and interface methods"""
        self.wdf: LocalDF = wdf
        return self

    def _gen_similarity_pairs(self, array: np.ndarray) -> Iterator[SimilarPairIndices]:
        del array  # Unused
        raise NotImplementedError

    def build_union_find(self: BaseStrategyProtocol, columns: Columns) -> tuple[UF, int]:
        self.validate(columns)
        array = self.wdf.get_array(columns, with_na=self.with_na_placeholder)

        n = len(array)

        uf = UnionFind(range(n))
        for i, j in self._gen_similarity_pairs(array):
            uf.union(i, j)

        return uf, n

    def canonicalizer(
        self,
        *,
        components: SingleComponents | MultiComponents,
        drop_duplicates: bool,
        keep: Keep,
    ) -> LocalDF:
        canonicals = self.wdf.get_canonical()

        n = len(canonicals)

        rep_index: dict[int, int] = {}
        for members in components.values():
            if keep == "first":
                rep = min(members)
            elif keep == "last":
                rep = max(members)

            for i in members:
                rep_index[i] = rep

        new_canonicals = np.array(
            [canonicals[rep_index[i]] for i in range(n)],
            dtype=object,
        )

        self.wdf.put_col(CANONICAL_ID, new_canonicals)

        if not drop_duplicates:
            return self.wdf
        return self.wdf.drop_duplicates(keep=keep)

    def str_representation(self, name: str) -> str:
        args = ", ".join(repr(a) for a in self._init_args)
        kwargs = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs.items())

        joined = ", ".join(filter(None, [args, kwargs]))
        return f"{name}({joined})"

    def __repr__(self):
        return self.str_representation(self.__class__.__name__)

    def __str__(self):
        # overridable; fall-back to:
        return self.__repr__()


class SingleColumnValidationMixin:
    """
    @private
    """

    def validate(self, columns: Columns) -> None:
        if not isinstance(columns, str):
            raise ValueError("For single column strategies, `columns` must be defined as a string")


class CompoundColumnValidationMixin:
    """
    @private
    """

    def validate(self, columns: Columns) -> None:
        if not isinstance(columns, tuple):
            raise ValueError("For compound columns strategies, `columns` must be defined as a tuple")


# EXACT DEDUPER:


@final
class Exact(BaseStrategy):
    """
    @private
    """

    NAME: str = "exact"

    @override
    def validate(self, columns):
        del columns  # Unused
        pass

    @override
    def _gen_similarity_pairs(self, array: np.ndarray):
        buckets = defaultdict(list)

        for i, v in enumerate(array):
            key = v if array.ndim == 1 else tuple(v.tolist())
            buckets[key].append(i)

        for indices in buckets.values():
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    yield indices[i], indices[j]

    def __str__(self):
        return self.str_representation(self.NAME)


# BINARY DEDUPERS:


class BinaryDedupers(BaseStrategy):
    """
    @private
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    def __invert__(self):
        return _NegatedBinaryDeduper(self)


class _NegatedBinaryDeduper(BinaryDedupers):
    """
    Internal strategy that negates another BinaryDedupers.
    """

    def __init__(self, inner: BinaryDedupers):
        self._inner = inner

    def _matches(self, value):
        return not self._inner._matches(value)

    def __str__(self):
        return f"~{self._inner}"

    def validate(self, columns):
        return getattr(self._inner, "validate")(columns)


@final
class IsNA(
    BinaryDedupers,
    SingleColumnValidationMixin,
):
    """
    @private
    Deduplicates all missing / null values into a single group.
    """

    NAME: str = "isna"

    with_na_placeholder: bool = False

    @override
    def _gen_similarity_pairs(self, array: np.ndarray):
        indices: list[int] = []

        for i, v in enumerate(array):
            # Spark & Polars
            if v is None:
                indices.append(i)
                continue

            if v is pd.NA:
                indices.append(i)
                continue  # important! next line would break otherwise.

            if v != v:
                indices.append(i)

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                yield indices[i], indices[j]

    def __str__(self):
        return self.str_representation(self.NAME)

    def __invert__(self):
        return _NotNA()


@final
class _NotNA(
    BaseStrategy,
    SingleColumnValidationMixin,
):
    """
    @private
    Deduplicate all non-NA / non-null values.
    """

    with_na_placeholder: bool = False

    @override
    def _gen_similarity_pairs(self, array: np.ndarray):
        indices: list[int] = []

        for i, v in enumerate(array):
            notna = True
            if v is None:
                notna = False
            if v is pd.NA:
                notna = False
            elif v != v:
                notna = False

            if notna:
                indices.append(i)

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                yield indices[i], indices[j]


@final
class StrLen(
    BinaryDedupers,
    SingleColumnValidationMixin,
):
    """
    TODO
    """

    NAME: str = "str_len"

    def __init__(self, min_len: int = 0, max_len: int | None = None):
        super().__init__(min_len=min_len, max_len=max_len)
        self._min_len = min_len
        self._max_len = max_len

    @override
    def _matches(self, value: str | None) -> bool:
        if not value:
            return False
        len_val = len(value)
        if not self._max_len:
            return len_val > self._min_len
        return len_val > self._min_len and len_val <= self._max_len

    def __str__(self):
        return self.str_representation(self.NAME)


@final
class StrStartsWith(
    BinaryDedupers,
    SingleColumnValidationMixin,
):
    """
    @private
    Strings start with canonicalizer.

    Defaults to case sensitive.

    Regex is not supported, please use `StrContains` otherwise.
    """

    NAME: str = "str_startswith"

    def __init__(self, pattern: str, case: bool = True):
        super().__init__(pattern=pattern, case=case)
        self._pattern = pattern
        self._case = case

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

    def __str__(self):
        return self.str_representation(self.NAME)


@final
class StrEndsWith(
    BinaryDedupers,
    SingleColumnValidationMixin,
):
    """
    @private
    Strings start with canonicalizer.

    Defaults to case sensitive.

    Regex is not supported, please use `StrContains` otherwise.
    """

    NAME: str = "str_endswith"

    def __init__(self, pattern: str, case: bool = True):
        super().__init__(pattern=pattern, case=case)
        self._pattern = pattern
        self._case = case

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

    def __str__(self):
        return self.str_representation(self.NAME)


@final
class StrContains(
    BinaryDedupers,
    SingleColumnValidationMixin,
):
    """
    @private
    Strings contains canonicalizer.

    Defaults to case sensitive. Supports literal substring or regex search.
    """

    NAME: str = "str_contains"

    def __init__(self, pattern: str, case: bool = True, regex: bool = False):
        super().__init__(pattern=pattern, case=case, regex=regex)
        self._pattern = pattern
        self._case = case
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

    def __str__(self):
        return self.str_representation(self.NAME)


# THRESHOLD DEDUPERS:


class ThresholdDedupers(BaseStrategy):
    """
    @private
    """

    def __init__(self, threshold: float = 0.95, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self._threshold = threshold

        if not (0 <= threshold < 1):
            raise ValueError("The threshold value must be greater or equal to 0 and less than 1")


@final
class Fuzzy(
    ThresholdDedupers,
    SingleColumnValidationMixin,
):
    """
    @private
    """

    NAME: str = "fuzzy"

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

    def __str__(self):
        return self.str_representation(self.NAME)


@final
class TfIdf(
    ThresholdDedupers,
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

    NAME: str = "tfidf"

    def __init__(
        self,
        ngram: int | tuple[int, int] = 3,
        threshold: float = 0.95,
        topn: int = 2,
        **kwargs,
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

    def __str__(self):
        return self.str_representation(self.NAME)


@final
class LSH(
    ThresholdDedupers,
    SingleColumnValidationMixin,
):
    """
    @private
    """

    NAME: str = "lsh"

    def __init__(
        self,
        ngram: int = 3,
        num_perm: int = 128,
        threshold: float = 0.95,
    ):
        super().__init__(
            threshold=threshold,
            ngram=ngram,
            num_perm=num_perm,
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

    def __str__(self):
        return self.str_representation(self.NAME)


# COMPOUND COLUMN:


@final
class Jaccard(
    ThresholdDedupers,
    CompoundColumnValidationMixin,
):
    """
    @private
    """

    NAME: str = "jaccard"

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

    def __str__(self):
        return self.str_representation(self.NAME)


@final
class Cosine(
    ThresholdDedupers,
    CompoundColumnValidationMixin,
):
    """
    @private
    """

    NAME: str = "cosine"

    def _gen_similarity_pairs(self, array: np.ndarray) -> Iterator[SimilarPairIndices]:
        n = len(array)
        for idx in range(n):
            for idy in range(idx + 1, n):
                arrx = array[idx]
                arry = array[idy]
                mask = [
                    (
                        x is not None
                        and y is not None
                        and not (isinstance(x, float) and np.isnan(x))
                        and not (isinstance(y, float) and np.isnan(y))
                    )
                    for x, y in zip(arrx, arry)
                ]
                arrx_masked = arrx[mask]
                arry_masked = arry[mask]
                product = np.dot(arrx_masked, arry_masked)

                if not product:
                    continue  # no match

                norms = norm(arrx_masked) * norm(arry_masked)

                if not norms:
                    continue  # zero div: guardrail

                if product / norms > self._threshold:
                    yield idx, idy

    def __str__(self):
        return self.str_representation(self.NAME)


# PUBLIC:


def exact() -> BaseStrategy:
    """TODO"""
    return Exact()


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
