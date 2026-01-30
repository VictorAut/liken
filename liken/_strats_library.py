"""Defines Deduplication strategies:

Strategies are either:
    - "Threshold" strategies: deduplication is decided according to a
        smiilarity. Routed through main package.
    - "Binary" strategies: deduplication is decided according to discrete
        outcomes. As this choice is fit for combinations using "and"
        operations, this is routed via the "rules" sub-package.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterator
from functools import cache
from typing import TYPE_CHECKING
from typing import Iterable
from typing import Protocol
from typing import Self
from typing import final

import numpy as np
import pandas as pd
from datasketch import MinHash
from datasketch import MinHashLSH
from networkx.utils.union_find import UnionFind
from numpy.linalg import norm
from rapidfuzz import fuzz
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from typing_extensions import override

from liken._constants import CANONICAL_ID


if TYPE_CHECKING:
    from liken._dataframe import LocalDF
    from liken._executors import MultiComponents
    from liken._executors import SingleComponents
    from liken._types import Columns
    from liken._types import Keep
    from liken._types import SimilarPairIndices


# INTERFACE:


class Base(Protocol):
    wdf: LocalDF
    with_na_placeholder: bool

    def set_frame(self, wdf: LocalDF) -> Self: ...
    def _gen_similarity_pairs(self, array: np.ndarray) -> Iterator[SimilarPairIndices]: ...
    def build_union_find(self, columns: Columns) -> tuple[UnionFind[int], int]: ...
    def canonicalizer(
        self,
        *,
        components: SingleComponents | MultiComponents,
        drop_duplicates: bool,
        keep: Keep,
    ) -> LocalDF: ...
    def str_representation(self, name: str) -> str: ...
    def validate(self, columns: Columns) -> None: ...


# BASE STRATEGY:


class BaseStrategy(Base):
    """
    Base Deduplication class
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

    def build_union_find(self: Base, columns: Columns) -> tuple[UnionFind[int], int]:
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


class SingleColumnMixin:
    """
    Validates the column type of deduplication strategy when passed in the
    columns arg. Only single strings allowed.
    """

    def validate(self, columns: Columns) -> None:
        if not isinstance(columns, str):
            raise ValueError("For single column strategies, `columns` must be defined as a string")


class CompoundValidationMixin:
    """
    Validates the column type of deduplication strategy when passed in the
    columns arg. Only tuples of strings allowed
    """

    def validate(self, columns: Columns) -> None:
        if not isinstance(columns, tuple):
            raise ValueError("For compound columns strategies, `columns` must be defined as a tuple")


# EXACT DEDUPER:


@final
class Exact(BaseStrategy):
    """
    Exact deduper.

    Does not accept a validation mixin (and therefore overrides validation)
    As the exact deduper can be applied to single, or compound columns.
    """

    name: str = "exact"

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
        return self.str_representation(self.name)


# BINARY DEDUPERS:


class BinaryDedupers(BaseStrategy):
    """
    Defines Binary "choice" deduplications, i.e. those that produce a discrete
    outcome. Any pair of values that satisfies the conditions of a Binary
    Deduper will be deduplicated.

    For example, if StrStartsWith is used for all strings starting with "a",
    then all records for the column starting with the character "a" will
    be canonicalised to the same record.
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
    Composable deduplication instance that inverts the results of any binary
    deduper (except IsNA deduper which follows it's own inversion logic).
    """

    def __init__(self, inner: BinaryDedupers):
        self._inner = inner

    def _matches(self, value):
        """simply return the inner classes opposed set of matches"""
        return not self._inner._matches(value)

    def __str__(self):
        return f"~{self._inner}"

    def validate(self, columns):
        "Get the inner instances validation mixin method"
        return getattr(self._inner, "validate")(columns)


@final
class IsNA(
    SingleColumnMixin,
    BinaryDedupers,
):
    """
    Deduplicates all missing / null values into a single group.

    Inversion operator here calls it's own negation class
    """

    name: str = "isna"

    # do NOT want to placehold Null values
    # As we are deduping on them and need to keep them to identify them
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
        return self.str_representation(self.name)

    def __invert__(self):
        return _NotNA()


@final
class _NotNA(
    SingleColumnMixin,
    BaseStrategy,  # TODO, is this correct? Should it not be BinaryDeduper for consistency?
):
    """
    Deduplicate all non-NA / non-null values.

    "not a match" for not null does not hold like it does for other Binary
    Dedupers.
    """

    name: str = "~isna"

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

    def __str__(self):
        return self.str_representation(self.name)


@final
class IsIn(
    SingleColumnMixin,
    BinaryDedupers,
):
    """
    Deduplicates all instances of strings that are a member of a defined
    iterable
    """

    name: str = "isin"

    def __init__(self, values: Iterable):
        super().__init__(values=values)
        self._values = values

    @override
    def _matches(self, value: str | None) -> bool:
        return value in self._values

    def __str__(self):
        return self.str_representation(self.name)


@final
class StrLen(
    SingleColumnMixin,
    BinaryDedupers,
):
    """
    Deduplicates all instances of strings that satisfy the bounds in
    (min_len, max_len) where the upper bound can actually be left unbounded.
    """

    name: str = "str_len"

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
        return self.str_representation(self.name)


@final
class StrStartsWith(
    SingleColumnMixin,
    BinaryDedupers,
):
    """
    Strings start with canonicalizer.

    Defaults to case sensitive.

    Regex is not supported, please use `StrContains` otherwise.
    """

    name: str = "str_startswith"

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
        return self.str_representation(self.name)


@final
class StrEndsWith(
    SingleColumnMixin,
    BinaryDedupers,
):
    """
    Strings start with canonicalizer.

    Defaults to case sensitive.

    Regex is not supported, please use `StrContains` otherwise.
    """

    name: str = "str_endswith"

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
        return self.str_representation(self.name)


@final
class StrContains(
    SingleColumnMixin,
    BinaryDedupers,
):
    """
    Strings contains canonicalizer.

    Defaults to case sensitive. Supports literal substring or regex search.
    """

    name: str = "str_contains"

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
        return self.str_representation(self.name)


# THRESHOLD DEDUPERS:


class ThresholdDedupers(BaseStrategy):
    """
    Base instance of dedupers that implement any similarity comparison
    mechanism.
    """

    def __init__(self, threshold: float = 0.95, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self._threshold = threshold

        if not (0 <= threshold < 1):
            raise ValueError("The threshold value must be greater or equal to 0 and less than 1")


@final
class Fuzzy(
    SingleColumnMixin,
    ThresholdDedupers,
):
    """
    Fuzzy string matching deduper
    """

    name: str = "fuzzy"

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
        return self.str_representation(self.name)


@final
class TfIdf(
    SingleColumnMixin,
    ThresholdDedupers,
):
    """
    TF-IDF deduper.

    Additional keywords arguments can be passed to parametrise the vectorizer,
    as listed in the [TF-IDF vectorizer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
    """

    name: str = "tfidf"

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
        return self.str_representation(self.name)


@final
class LSH(
    SingleColumnMixin,
    ThresholdDedupers,
):
    """
    Locality Sensitive Hashing deduper
    """

    name: str = "lsh"

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
        return self.str_representation(self.name)


# COMPOUND COLUMN:


@final
class Jaccard(
    CompoundValidationMixin,
    ThresholdDedupers,
):
    """
    Deduplicate sets where such sets contain categorical data.
    """

    name: str = "jaccard"

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
        return self.str_representation(self.name)


@final
class Cosine(
    CompoundValidationMixin,
    ThresholdDedupers,
):
    """
    Deduplicate sets where such sets contain numeric data.
    """

    name: str = "cosine"

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
        return self.str_representation(self.name)


# PUBLIC PKG:


def exact() -> BaseStrategy:
    """Exact Deduplication.

    Can deduplicate a single column, or multiple columns.

    If no strategies are applied to `Dedupe`, `exact` is applied by default.

    Returns:
        Instance of `BaseStrategy`..

    Example:
        Applied to a single column:

            from liken import Dedupe, exact

            lk = Dedupe(df)
            lk.apply(exact())
            df = lk.drop_duplicates("address")

        Applied to multiple columns:

            lk = Dedupe(df)
            lk.apply(exact())
            df = lk.drop_duplicates(("address", "email"))

        E.g.

            >>> df # Before
            +------+-----------+--------------------+
            | id   |  address  |        email       |
            +------+-----------+--------------------+
            |  1   |  london   |  fizzpop@gmail.com |
            |  2   |   null    |  foobar@gmail.com  |
            |  3   |   null    |  foobar@gmail.com  |
            +------+-----------+--------------------+

            >>> df # After
            +------+-----------+---------------------+
            | id   |  address  |        email        |
            +------+-----------+---------------------+
            |  1   |  london   |  fizzpop@gmail.com  |
            |  2   |   null    |  foobar@gmail.com   |
            +------+-----------+---------------------+

        By default `exact` is used when no stratgies are explicitely applied:

            lk = Dedupe(df)
            lk.drop_duplicates("address")   # OK, still dedupes.
    """
    return Exact()


def fuzzy(threshold: float = 0.95) -> BaseStrategy:
    """Near string deduplication.

    Usage is on single columns of a dataframe.

    Args:
        threshold: the minimum threshold at which similarity between two pairs
            of values will be considered valid for deduplication.

    Returns:
        Instance of `BaseStrategy`.

    Example:
        Applied to a single column:

            from liken import Dedupe, fuzzy

            lk = Dedupe(df)
            lk.apply({"address": fuzzy(threshold=0.8)})
            df = lk.drop_duplicates(keep="last")

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
    """
    return Fuzzy(threshold=threshold)


def tfidf(
    threshold: float = 0.95,
    ngram: int | tuple[int, int] = 3,
    topn: int = 2,
    **kwargs,
) -> BaseStrategy:
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
        Instance of `BaseStrategy`.

    Example:
        Applied to a single column:

            from liken import Dedupe, tfidf

            lk = Dedupe(df)
            lk.apply({"address": tfidf(threshold=0.8, ngram=1)})
            df = lk.drop_duplicates(keep="last")

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


def lsh(
    threshold: float = 0.95,
    ngram: int = 3,
    num_perm: int = 128,
) -> BaseStrategy:
    """Near string deduplication using locality sensitive hashing (LSH).

    Usage is on single columns of a dataframe. `lsh` is a tuneable deduper.
    Experimentation is required for optimal use.

    Args:
        threshold: the minimum threshold at which similarity between two pairs
            of values will be considered valid for deduplication.
        ngram: the number of character ngrams to consider. For `lsh`, and
            unlike the `tfidf` implementation, this is single integer ngram
            number. So, `ngram=1` is only unigrams. Increasing ngrams reduces
            overall deduplication. However, too small an `ngram` may result in false
            positives.
        num_perm: the number of MinHash permutations used to approximate
            similarity. Increasing this generally produces better matches, at
            greater computational cost. Very low numbers of permutations (< 64)
            can produce unreliable results.

    Returns:
        Instance of `BaseStrategy`.

    Example:
        Applied to a single column:

            from liken import Dedupe, lsh

            lk = Dedupe(df)
            lk.apply({"address": lsh(threshold=0.8, ngram=1)})
            df = lk.drop_duplicates(keep="last")

        E.g.

            >>> df
            +------+-----------+----------------------+
            | id   |  address  |         email        |
            +------+-----------+----------------------+
            |  1   |  london   |  fizzpop@gmail.com   |
            |  2   |   null    |  foobar@gmail.com    |
            |  3   |  london   |  foobar@gmail.co.uk  |
            +------+-----------+----------------------+

            >>> df # After deduplication
            +------+-----------+----------------------+
            | id   |  address  |         email        |
            +------+-----------+----------------------+
            |  1   |  london   |  fizzpop@gmail.com   |
            |  3   |  london   |  foobar@gmail.co.uk  |
            +------+-----------+----------------------+
    """
    return LSH(threshold=threshold, ngram=ngram, num_perm=num_perm)


def jaccard(threshold: float = 0.95) -> BaseStrategy:
    """Multi-column deduplication using jaccard similarity.

    Usage is on multiple columns of a dataframe. Appropriate for categorical
    data. Null types are handled out-of-box with jaccard, they are simply
    considered another category of a given field.

    Args:
        threshold: the minimum threshold at which similarity between two pairs
            of values will be considered valid for deduplication.

    Returns:
        Instance of `BaseStrategy`.

    Example:
        Applied to multiple columns:

            from liken import Dedupe, jaccard

            lk = Dedupe(df)
            lk.apply(jaccard())
            df = lk.drop_duplicates(
                ("account", "status", "country", "property"),
                keep="first",
            )

        E.g.

            >>> df
            +------+-----------+----------+----------+-----------+
            | id   |  account  |  status  |  country |  property |
            +------+-----------+----------+----------+-----------+
            |  1   |  reddit   |  married |    UK    |  house    |
            |  2   |  flickr   |  married |    UK    |  house    |
            |  3   | pinterest |  single  |  Germany |  flat     |
            +------+-----------+----------+----------+-----------+

            >>> df # After deduplication
            +------+-----------+----------+----------+-----------+
            | id   |  account  |  status  |  country |  property |
            +------+-----------+----------+----------+-----------+
            |  1   |  reddit   |  married |    UK    |  house    |
            |  3   | pinterest |  single  |  Germany |  flat     |
            +------+-----------+----------+----------+-----------+
    """
    return Jaccard(threshold=threshold)


def cosine(threshold: float = 0.95) -> BaseStrategy:
    """Multi-column deduplication using cosine similarity.

    Usage is on multiple columns of a dataframe. Appropriate for numerical
    data.

    Args:
        threshold: the minimum threshold at which similarity between two pairs
            of values will be considered valid for deduplication.

    Returns:
        Instance of `BaseStrategy`.

    Note:
        In the case of null types, that column is ignore, and only the
        similarity is taken of the remaining columns is taken.

        So, if deduplicating columns `col_1`, `col_2` and `col_3` with `cosine`,
        any similarity is usually the dot product for a given pairwise evaluation
        i.e.

            (`col_1i`, `col_2i`, `col_3i`) . (`col_1j`, `col_2j`, `col_3j`)

        However, if `col_1i` is Null then the following is evaluated:

            (`col_2i`, `col_3i`) . (`col_2j`, `col_3j`)

        Additionally, if `col_j2` is *also* Null then the following is evaluated:

            (`col_3i`) . (`col_3j`)

        Taking this into account you may find it best to avoid cosine similarity
        calculations for sparse datasets. Alternatively, you may opt to your
        approach by either preprocessing the Nulls beforehand, or, by
        limiting yourself to using the `cosine` deduplicator with the `Rules`
        API using combinations for non null fields, e.g.

            STRAT = Rules(
                on(
                    ("col_1", "col_2", "col_3"),
                    cosine(),
                )
                #
                & on("col_1", ~isna()),
            )

    Warning:
        Normalization is a standard approach to ensure that the results of
        cosine similarity are valid. Consider [standard
        approaches](https://scikit-learn.org/stable/modules/preprocessing.html#normalization)

    Example:
        Applied to multiple columns:

            from liken import Dedupe, cosine

            lk = Dedupe(df)
            lk.apply(cosine())
            df = lk.drop_duplicates(
                ("surface are", "ceiling height", "building age", "num_rooms"),
                keep="first",
            )
    """
    return Cosine(threshold=threshold)


# RULES SUB PKG


def isna() -> BaseStrategy:
    """Discrete deduper on null/None values.

    Usage is on a single column of a dataframe. Available as the inversion, i.e.
    "not null" using inversion operator: `~isna()`.

    Returns:
        Instance of `BaseStrategy`.

    Example:
        Applied to a single column:

            from liken import Dedupe, exact
            from liken.rules import Rules, on, isna

            STRAT = Rules(on("email", exact()) & on("address", ~isna()))

            lk = Dedupe(df)
            lk.apply(STRAT)
            df = lk.drop_duplicates(keep="last")

            >>> df # before
            +------+-----------+---------------------+
            | id   |  address  |         email       |
            +------+-----------+---------------------+
            |  1   |  london   |  fizzpop@yahoo.com  |
            |  2   |  london   |  fizzpop@yahoo.com  |
            |  3   |   null    |  foobar@gmail.com   |
            |  4   |   null    |  foobar@gmail.com   |
            +------+-----------+---------------------+

            >>> df # after
            +------+-----------+---------------------+
            | id   |  address  |         email       |
            +------+-----------+---------------------+
            |  2   |  london   |  fizzpop@yahoo.com  |
            |  3   |   null    |  foobar@gmail.com   |
            |  4   |   null    |  foobar@gmail.com   | # Not deduped!
            +------+-----------+---------------------+
    """
    return IsNA()


def isin(values: Iterable) -> BaseStrategy:
    """Discrete deduper for membership testing.

    Usage is on a single column of a dataframe. Available as the inversion, i.e.
    "not in" using inversion operator: `~isin()`.

    Returns:
        Instance of `BaseStrategy`.

    Example:
        Applied to a single column:

            from liken import Dedupe, exact
            from liken.rules import Rules, on, isin

            STRAT = Rules(on("address", isin(values="london")))

            lk = Dedupe(df)
            lk.apply(STRAT)
            df = lk.drop_duplicates(keep="last")

            >>> df # before
            +------+-----------+---------------------+
            | id   |  address  |         email       |
            +------+-----------+---------------------+
            |  1   |  london   |  fizzpop@yahoo.com  |
            |  2   |  london   |   hello@yahoo.com   |
            |  3   |   null    |  foobar@gmail.com   |
            |  4   |   null    |  random@gmail.com   |
            |  5   |  london   |  butterfly@msn.jp   |
            +------+-----------+---------------------+

            >>> df # after
            +------+-----------+---------------------+
            | id   |  address  |         email       |
            +------+-----------+---------------------+
            |  3   |   null    |  foobar@gmail.com   |
            |  4   |   null    |  random@gmail.com   |
            |  5   |  london   |  butterfly@msn.jp   |
            +------+-----------+---------------------+
    """
    return IsIn(values=values)


def str_len(min_len: int = 0, max_len: int | None = None) -> BaseStrategy:
    """Discrete deduper on string length.

    Usage is on a single column of a dataframe. Available as the inversion, i.e.
    "not the defined length" using inversion operator: `~str_len()`.

    Deduplication will happen over the bounded lengths defined by `min_len` and
    `max_len`. The upper end of the range can be left unbounded. For
    deduplication over an exact length use `max_len = min_len + 1`.

    Args:
        min_len: the lower bound of lengths considered
        max_len: the upper bound of lengths considered. Can be left unbounded.

    Returns:
        Instance of `BaseStrategy`.

    Example:
        Applied to a single column:

            from liken import Dedupe, exact
            from liken.rules import Rules, on, isna

            STRAT = Rules(on("email", exact()) & on("email", str_len(min_len=10)))

            lk = Dedupe(df)
            lk.apply(STRAT)
            df = lk.drop_duplicates(keep="last")

            >>> df # before
            +------+-----------+---------------------+
            | id   |  address  |         email       |
            +------+-----------+---------------------+
            |  1   |  london   |  fizzpop@yahoo.com  |
            |  2   |   tokyo   |  fizzpop@yahoo.com  |
            |  3   |   paris   |       a@msn.fr      |
            |  4   |   nice    |       a@msn.fr      |
            +------+-----------+---------------------+

            >>> df # after
            +------+-----------+---------------------+
            | id   |  address  |         email       |
            +------+-----------+---------------------+
            |  2   |   tokyo   |  fizzpop@yahoo.com  |
            |  3   |   paris   |       a@msn.fr      |
            |  4   |   nice    |       a@msn.fr      |
            +------+-----------+---------------------+
    """
    return StrLen(min_len=min_len, max_len=max_len)


def str_startswith(pattern: str, case: bool = True) -> BaseStrategy:
    """Discrete deduper on strings starting with a pattern.

    Usage is on a single column of a dataframe. Available as the inversion, i.e.
    "not starting with pattern" using inversion operator: `~str_startswith()`.

    Deduplication will happen for any pairwise matches that have the same
    `pattern`. Case sensitive unless optionally removed.

    Args:
        pattern: the pattern that the string starts with to be deduplicated
        case: case sensitive, or not.

    Returns:
        Instance of `BaseStrategy`.

    Example:
        Applied to a single column:

            from liken import Dedupe, exact
            from liken.rules import Rules, on, str_startswith

            STRAT = Rules(
                on("email", exact())
                & on(
                    "email",
                    str_startswith(pattern="f", case=True),
                )
            )

            lk = Dedupe(df)
            lk.apply(STRAT)
            df = lk.drop_duplicates(keep="first")

            >>> df
            +------+-----------+---------------------+
            | id   |  address  |         email       |
            +------+-----------+---------------------+
            |  1   | new york  |  fizzpop@yahoo.com  |
            |  2   |   london  |  foobar@gmail.co.uk |
            |  3   | marseille |   Flipflop@msn.fr   |
            |  4   |  chicago  |    random@aol.com   |
            +------+-----------+---------------------+

            >>> df # after
            +------+-----------+---------------------+
            | id   |  address  |         email       |
            +------+-----------+---------------------+
            |  1   | new york  |  fizzpop@yahoo.com  |
            |  3   | marseille |   Flipflop@msn.fr   |
            |  4   |  chicago  |    random@aol.com   |
            +------+-----------+---------------------+
    """
    return StrStartsWith(pattern=pattern, case=case)


def str_endswith(pattern: str, case: bool = True) -> BaseStrategy:
    """Discrete deduper on strings ending with a pattern.

    Usage is on a single column of a dataframe. Available as the inversion, i.e.
    "not ending with pattern" using inversion operator: `~str_endswith()`.

    Deduplication will happen for any pairwise matches that have the same
    `pattern`. Case sensitive unless optionally removed.

    Args:
        pattern: the pattern that the string ends with to be deduplicated
        case: case sensitive, or not.

    Returns:
        Instance of `BaseStrategy`.

    Example:
        Applied to a single column:

            from liken import Dedupe, exact
            from liken.rules import Rules, on, str_endswith

            STRAT = Rules(
                on("email", exact())
                & on(
                    "email",
                    str_endswith(pattern=".com", case=False),
                )
            )

            lk = Dedupe(df)
            lk.apply(STRAT)
            df = lk.drop_duplicates(keep="first")

            >>> df
            +------+-----------+---------------------+
            | id   |  address  |         email       |
            +------+-----------+---------------------+
            |  1   | new york  |  fizzpop@yahoo.Com  |
            |  2   |   london  |  foobar@gmail.co.uk |
            |  3   | marseille |   Flipflop@msn.fr   |
            |  4   |  chicago  |    random@aol.com   |
            +------+-----------+---------------------+

            >>> df # after
            +------+-----------+---------------------+
            | id   |  address  |         email       |
            +------+-----------+---------------------+
            |  1   | new york  |  fizzpop@yahoo.Com  |
            |  2   |   london  |  foobar@gmail.co.uk |
            |  3   | marseille |   Flipflop@msn.fr   |
            +------+-----------+---------------------+
    """
    return StrEndsWith(pattern=pattern, case=case)


def str_contains(
    pattern: str,
    case: bool = True,
    regex: bool = False,
) -> BaseStrategy:
    """Discrete deduper on general string patterns with regex.

    Usage is on a single column of a dataframe. Available as the inversion, i.e.
    "not containing pattern" using inversion operator: `~str_contains()`.

    Deduplication will happen for any pairwise matches that have the same
    `pattern`. Case sensitive unless optionally removed. Pattern can include
    regex patterns if passed with `regex` arg.

    Args:
        pattern: the pattern that the string ends with to be deduplicated
        case: case sensitive, or not.
        regex: uses regex patterns, or not.

    Returns:
        Instance of `BaseStrategy`.

    Example:
        Applied to a single column:

            from liken import Dedupe, exact
            from liken.rules import Rules, on, str_contains

            STRAT = Rules(
                on("email", exact())
                & on(
                    "email",
                    str_contains(pattern=r"05\d{3}", regex=True),
                )
            )

            lk = Dedupe(df)
            lk.apply(STRAT)
            df = lk.canonicalize(keep="first")

            >>> df
            +------+-----------------------------+
            | id   |           address           |
            +------+-----------------------------+
            |  1   | 12 calle girona, 05891, ES  |
            |  2   |  1A avenida palmas, 05562   |
            |  3   |      901, Spain, 05435      |
            |  4   |     12, santiago, 09945     |
            +------+-----------------------------+

            >>> df # after
            +------+-----------------------------+---------------+
            | id   |           address           |  canonical_id |
            +------+-----------------------------+---------------+
            |  1   | 12 calle girona, 05891, ES  |        1      |
            |  2   |  1A avenida palmas, 05562   |        1      |
            |  3   |      901, Spain, 05435      |        1      |
            |  4   |     12, santiago, 09945     |        4      |
            +------+-----------------------------+---------------+
    """
    return StrContains(pattern=pattern, case=case, regex=regex)
