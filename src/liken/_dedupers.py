"""Defines Deduplication classes:

E.g. "fuzzy"

Dedupers are either:
    - "Threshold" dedupers: deduplication is decided according to a
        smiilarity. Routed through main package.
    - "Predicate" dedupers: deduplication is decided according to discrete
        outcomes. As this choice is fit for combinations using "and"
        operations, this is routed via the "rules" module.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterator
from typing import TYPE_CHECKING
from typing import Callable
from typing import ClassVar
from typing import Iterable
from typing import Literal
from typing import Protocol
from typing import Self
from typing import final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from datasketch import MinHash
from datasketch import MinHashLSH
from networkx.utils.union_find import UnionFind
from rapidfuzz import fuzz
from rapidfuzz import process
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from typing_extensions import override

from liken._constants import CANONICAL_ID
from liken._registry import registry
from liken._preprocessors import Preprocessor


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
    def _gen_similarity_pairs(self, array: pa.Array | pa.Table) -> Iterator[SimilarPairIndices]: ...
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
    def preprocess() -> pa.Array | pa.Table: ...


# BASE DEDUPER:


class BaseDeduper(Base):
    """
    Base Deduplication class

    By default all dedupers will operate on filled nulls, thus treating them
    as identical instances within a column(s) of values,
    """

    with_na_placeholder: bool = True

    def __init__(self, *args, **kwargs):
        self._init_args = args
        self._init_kwargs = kwargs

    def set_frame(self, wdf: LocalDF) -> Self:
        """Inject dataframe and interface methods"""
        self.wdf: LocalDF = wdf
        return self

    def _gen_similarity_pairs(self, array: pa.Array | pa.Table) -> Iterator[SimilarPairIndices]:
        del array  # Unused
        raise NotImplementedError
    
    @staticmethod
    def preprocess(array: pa.Array | pa.Table, preprocessors: list[Preprocessor]) -> pa.Array | pa.Table:
        """apply a sequence of preprocessors"""
        if isinstance(array, pa.Table):
            return array
        for processor in preprocessors:
            processor.from_array(array)
            array = processor.process()
        return array

    def build_union_find(
        self: Base,
        columns: Columns,
        preprocessors: list[Preprocessor],
        predicate: set = set(),
    ) -> tuple[UnionFind[int], int]:
        self.validate(columns)

        array: pa.Array | pa.Table = self.wdf.get_array(columns, with_na=self.with_na_placeholder)

        array: pa.Array | pa.Table = self.preprocess(array, preprocessors)
        print(array)

        if predicate:
            # subsets the array on predicate indice list
            array: pa.Array | pa.Table = array.take(sorted(predicate))

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

            for member in members:
                rep_index[member] = rep

        # for predicated components, default to ith index for non deduplicated rows
        new_canonicals: list = [canonicals[rep_index.get(i, i)].as_py() for i in range(n)]

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
    Validates the type of `columns` param as passed to the deduper.
    Only single strings allowed.
    """

    def validate(self, columns: Columns) -> None:
        if not isinstance(columns, str):
            raise ValueError("For single column dedupers, `columns` must be defined as a string")


class CompoundColumnMixin:
    """
    Validates the column of of `columns` param as passed to the deduper.
    Only tuples of strings allowed.
    """

    def validate(self, columns: Columns) -> None:
        if not isinstance(columns, tuple):
            raise ValueError("For compound columns dedupers, `columns` must be defined as a tuple")


# EXACT DEDUPER:


@final
class Exact(BaseDeduper):
    """
    Exact deduper.

    Does not accept a validation mixin (and therefore overrides validation)
    As the exact deduper can be applied to single, or compound columns.
    """

    _NAME: ClassVar[str] = "exact"

    @override
    def validate(self, columns):
        del columns  # Unused
        pass

    @override
    def _gen_similarity_pairs(self, array: pa.Array | pa.Table):
        buckets = defaultdict(list)

        # single column
        if isinstance(array, pa.Array):
            for i, key in enumerate(array):
                buckets[key].append(i)

        # multi column
        if isinstance(array, pa.Table):
            columns = [array[col] for col in array.column_names]

            n = array.num_rows

            for i in range(n):
                key = tuple(col[i].as_py() for col in columns)
                buckets[key].append(i)

        for indices in buckets.values():
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    yield indices[i], indices[j]

    def __str__(self):
        return self.str_representation(self._NAME)


# PREDICATE DEDUPERS:


class PredicateDedupers(BaseDeduper):
    """
    Defines predicate, "choice", deduplications, i.e. those that produce a discrete
    outcome. Any pair of values that satisfies the conditions of a predicate
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

    def _vectorized_matches(self, array: pa.Array) -> pa.Array | None:
        """
        Optional vectorised implementation.

        Should return a boolean Arrow array if supported, otherwise None.
        """
        return None

    @override
    def _gen_similarity_pairs(self, array: pa.Array) -> Iterator[SimilarPairIndices]:

        # try vectorized approach, if available

        mask: pa.Array | None = self._vectorized_matches(array)

        if mask:
            indices = pc.indices_nonzero(mask).to_pylist()

            n = len(indices)
            for i in range(n):
                for j in range(i + 1, n):
                    yield indices[i], indices[j]
            return

        # fallback to non vectorized, i.e. "Python" matching:

        array: list = array.to_pylist()

        n = len(array)
        for i in range(n):
            if not self._matches(array[i]):
                continue
            for j in range(i + 1, n):
                if self._matches(array[j]):
                    yield i, j

    def __invert__(self):
        return _NegatedPredicateDeduper(self)


class _NegatedPredicateDeduper(PredicateDedupers):
    """
    Composable deduplication instance that inverts the results of any predicate
    deduper (except IsNA deduper which follows it's own inversion logic).
    """

    def __init__(self, inner: PredicateDedupers):
        self._inner = inner

    def _matches(self, value):
        """simply return the inner classes opposed set of matches"""
        return not self._inner._matches(value)

    def _vectorized_matches(self, array: pa.Array) -> pa.Array | None:
        """
        invert the resulting mask for a match.
        """
        mask = self._inner._vectorized_matches(array)

        if mask is None:
            return None

        return pc.invert(mask)

    def __str__(self):
        return f"~{self._inner}"

    def validate(self, columns):
        "Get the inner instances validation mixin method"
        return getattr(self._inner, "validate")(columns)


@final
class IsNA(
    SingleColumnMixin,
    PredicateDedupers,
):
    """
    Deduplicates all missing / null values into a single group.

    Inversion operator here calls it's own negation class
    """

    _NAME: ClassVar[str] = "isna"

    # do NOT want to placehold Null values
    # As we are deduping on them and need to keep them to identify them
    with_na_placeholder: bool = False

    @override
    def _gen_similarity_pairs(self, array: pa.Array):
        array: list = array.to_pylist()

        indices: list[int] = []

        for i, v in enumerate(array):
            if v is None:
                indices.append(i)
                continue

            if v != v:
                indices.append(i)

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                yield indices[i], indices[j]

    def __str__(self):
        return self.str_representation(self._NAME)

    def __invert__(self):
        return _NotNA()


@final
class _NotNA(
    SingleColumnMixin,
    PredicateDedupers,
):
    """
    Deduplicate all non-NA / non-null values.

    "not a match" for not null does not hold like it does for other predicate
    Dedupers.
    """

    _NAME: ClassVar[str] = "~isna"

    with_na_placeholder: bool = False

    @override
    def _gen_similarity_pairs(self, array: pa.Array):
        array: list = array.to_pylist()

        indices: list[int] = []

        for i, v in enumerate(array):
            notna = True
            if v is None:
                notna = False

            elif v != v:
                notna = False

            if notna:
                indices.append(i)

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                yield indices[i], indices[j]

    def __str__(self):
        return self.str_representation(self._NAME)


@final
class IsIn(
    SingleColumnMixin,
    PredicateDedupers,
):
    """
    Deduplicates all instances of strings that are a member of a defined
    iterable
    """

    _NAME: ClassVar[str] = "isin"

    def __init__(self, values: Iterable):
        super().__init__(values=values)
        self._values = values

    @override
    def _matches(self, value: str | None) -> bool:
        return value in self._values

    def __str__(self):
        return self.str_representation(self._NAME)


@final
class StrLen(
    SingleColumnMixin,
    PredicateDedupers,
):
    """
    Deduplicates all instances of strings that satisfy the bounds in
    (min_len, max_len) where the upper bound can actually be left unbounded.
    """

    _NAME: ClassVar[str] = "str_len"

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
        return self.str_representation(self._NAME)


@final
class StrStartsWith(
    SingleColumnMixin,
    PredicateDedupers,
):
    """
    Strings start with canonicalizer.

    Defaults to case sensitive.

    Regex is not supported, please use `StrContains` otherwise.
    """

    _NAME: ClassVar[str] = "str_startswith"

    def __init__(self, pattern: str, case: bool = True):
        super().__init__(pattern=pattern, case=case)
        self._pattern = pattern
        self._case = case

    @override
    def _vectorized_matches(self, array: pa.Array) -> pa.Array:

        if self._case:
            return pc.starts_with(array, self._pattern)

        return pc.starts_with(
            pc.utf8_lower(array),
            self._pattern.lower(),
        )

    def __str__(self):
        return self.str_representation(self._NAME)


@final
class StrEndsWith(
    SingleColumnMixin,
    PredicateDedupers,
):
    """
    Strings start with canonicalizer.

    Defaults to case sensitive.

    Regex is not supported, please use `StrContains` otherwise.
    """

    _NAME: ClassVar[str] = "str_endswith"

    def __init__(self, pattern: str, case: bool = True):
        super().__init__(pattern=pattern, case=case)
        self._pattern = pattern
        self._case = case

    @override
    def _vectorized_matches(self, array: pa.Array) -> pa.Array:

        if self._case:
            return pc.ends_with(array, self._pattern)

        return pc.ends_with(
            pc.utf8_lower(array),
            self._pattern.lower(),
        )

    def __str__(self):
        return self.str_representation(self._NAME)


@final
class StrContains(
    SingleColumnMixin,
    PredicateDedupers,
):
    """
    Strings contains canonicalizer.

    Defaults to case sensitive. Supports literal substring or regex search.
    """

    _NAME: ClassVar[str] = "str_contains"

    def __init__(self, pattern: str, case: bool = True, regex: bool = False):
        super().__init__(pattern=pattern, case=case, regex=regex)
        self._pattern = pattern
        self._case = case
        self._regex = regex

        if self._regex:
            flags = 0 if self._case else re.IGNORECASE
            self._compiled_pattern = re.compile(self._pattern, flags)

    @override
    def _vectorized_matches(self, array: pa.Array) -> pa.Array:

        if self._regex:
            if self._case:
                return pc.match_substring_regex(array, self._pattern)
            else:
                return pc.match_substring_regex(array, self._pattern, ignore_case=True)

        if self._case:
            return pc.match_substring(array, self._pattern)

        return pc.match_substring(
            pc.utf8_lower(array),
            self._pattern.lower(),
        )

    def __str__(self):
        return self.str_representation(self._NAME)


# THRESHOLD DEDUPERS:


class ThresholdDedupers(BaseDeduper):
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

    _NAME: ClassVar[str] = "fuzzy"

    _SCORERS: dict[str, Callable] = {
        "simple_ratio": fuzz.ratio,
        "partial_ratio": fuzz.partial_ratio,
        "token_sort_ratio": fuzz.token_sort_ratio,
        "token_set_ratio": fuzz.token_set_ratio,
        "weighted_ratio": fuzz.WRatio,
        "quick_ratio": fuzz.QRatio,
    }

    def __init__(
        self,
        threshold: float = 0.95,
        scorer: Literal[
            "simple_ratio",
            "partial_ratio",
            "token_sort_ratio",
            "token_set_ratio",
            "weighted_ratio",
            "quick_ratio",
        ] = "simple_ratio",
    ):
        super().__init__(
            threshold=threshold,
            scorer=scorer,
        )
        self._threshold = threshold
        self._scorer = scorer

    def get_scorer(self):
        return self._SCORERS.get(self._scorer, fuzz.ratio)

    def _gen_similarity_pairs(self, array: pa.Array) -> Iterator[SimilarPairIndices]:
        array: list = array.to_pylist()
        n = len(array)

        threshold = 100 * self._threshold

        scorer = self.get_scorer()

        for i, s1 in enumerate(array):
            if i + 1 >= n:
                break

            scores = process.cdist(
                [s1],
                array[i + 1 :],
                scorer=scorer,
            )[0]

            for offset, score in enumerate(scores):
                if score > threshold:
                    yield i, i + 1 + offset

    def __str__(self):
        return self.str_representation(self._NAME)


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

    _NAME: ClassVar[str] = "tfidf"

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


@final
class LSH(
    SingleColumnMixin,
    ThresholdDedupers,
):
    """
    Locality Sensitive Hashing deduper
    """

    _NAME: ClassVar[str] = "lsh"

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

    def _gen_token(self, text: pa.Scalar) -> Iterator:
        for i in range(len(text) - self._ngram + 1):
            yield text[i : i + self._ngram]

    def _build_minhashes(self, array: list) -> list[MinHash]:
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

    def _gen_similarity_pairs(self, array: pa.Array) -> Iterator[SimilarPairIndices]:
        array: list = array.to_pylist()

        minhashes: list[MinHash] = self._build_minhashes(array)
        lsh: MinHashLSH = self._lsh(minhashes)

        for idx, minhash in enumerate(minhashes):
            for idy in lsh.query(minhash):
                if idx < idy:
                    yield idx, idy

    def __str__(self):
        return self.str_representation(self._NAME)


# COMPOUND COLUMN:


@final
class Jaccard(
    CompoundColumnMixin,
    ThresholdDedupers,
):
    """
    Deduplicate sets where such sets contain categorical data.
    """

    _NAME: ClassVar[str] = "jaccard"

    def _gen_similarity_pairs(self, array: pa.Table) -> Iterator[SimilarPairIndices]:
        columns = [array[col] for col in array.column_names]
        n = array.num_rows

        # Build row sets directly
        sets = [{col[i].as_py() for col in columns if col[i].as_py() is not None} for i in range(n)]

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
        return self.str_representation(self._NAME)


@final
class Cosine(
    CompoundColumnMixin,
    ThresholdDedupers,
):
    """
    Deduplicate sets where such sets contain numeric data.
    """

    _NAME: ClassVar[str] = "cosine"

    @override
    def _gen_similarity_pairs(self, array: pa.Table) -> Iterator[SimilarPairIndices]:

        columns = [array[col].to_numpy(zero_copy_only=False) for col in array.column_names]
        matrix = np.column_stack(columns)

        matrix = np.nan_to_num(matrix, nan=0.0)

        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1

        normalized = matrix / norms[:, None]

        n = normalized.shape[0]

        for i in range(n):
            sims = normalized[i] @ normalized[i + 1 :].T

            for offset, val in enumerate(sims):
                if val > self._threshold:
                    yield i, i + 1 + offset

    def __str__(self):
        return self.str_representation(self._NAME)


# PUBLIC PKG:


@registry.register("exact")
def exact() -> BaseDeduper:
    """Exact Deduplication.

    Can deduplicate a single column, or multiple columns.

    If no dedupers are applied to `Dedupe`, `exact` is applied by default.

    Returns:
        Instance of `BaseDeduper`..

    Example:
        Applied to a single column:

            import liken as lk

            df = (
                lk.dedupe(df)
                .apply(exact())
                .drop_duplicates("address")
                .collect()
            )

        Applied to multiple columns:

            df = (
                lk.dedupe(df)
                .apply(exact())
                .drop_duplicates(("address", "email"))
                .collect()
            )

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

        By default `exact` is used when no dedupers are explicitely applied:

            # OK, still dedupes.
            df = Dedupe(df).drop_duplicates("address").collect()
    """
    return Exact()


@registry.register("fuzzy")
def fuzzy(threshold: float = 0.95, scorer="simple_ratio") -> BaseDeduper:
    """Near string deduplication.

    Usage is on single columns of a dataframe.

    Args:
        threshold: The minimum threshold at which similarity between two pairs
            of values will be considered valid for deduplication.
        scorer: The fuzzy scorer. Defaults to "simple ratio". Options are
            "simple_ratio", "partial_ratio", "token_sort_ratio",
            "token_set_ratio", "weighted_ratio", "quick_ratio".

    Returns:
        Instance of `BaseDeduper`.

    Example:
        Applied to a single column:

            import liken as lk

            df = (
                lk.dedupe(df)
                .apply({"address": fuzzy(threshold=0.8)})
                .drop_duplicates(keep="last")
                .collect()
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
    """
    return Fuzzy(threshold=threshold, scorer=scorer)


@registry.register("tfidf")
def tfidf(
    threshold: float = 0.95,
    ngram: int | tuple[int, int] = 3,
    topn: int = 2,
    **kwargs,
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
                .collect()
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


@registry.register("lsh")
def lsh(
    threshold: float = 0.95,
    ngram: int = 3,
    num_perm: int = 128,
) -> BaseDeduper:
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
        Instance of `BaseDeduper`.

    Example:
        Applied to a single column:

            import liken as lk

            df = (
                lk.dedupe(df)
                .apply({"address": lsh(threshold=0.8, ngram=1)})
                .drop_duplicates(keep="last")
                .collect()
            )

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


@registry.register("jaccard")
def jaccard(threshold: float = 0.95) -> BaseDeduper:
    """Multi-column deduplication using jaccard similarity.

    Usage is on multiple columns of a dataframe. Appropriate for categorical
    data. Null types are handled out-of-box with jaccard, they are simply
    considered another category of a given field.

    Args:
        threshold: the minimum threshold at which similarity between two pairs
            of values will be considered valid for deduplication.

    Returns:
        Instance of `BaseDeduper`.

    Example:
        Applied to multiple columns:

            import liken as lk

            df = (
                lk.dedupe(df)
                .apply(jaccard())
                .drop_duplicates(
                    ("account", "status", "country", "property"),
                    keep="first",
                )
                .collect()
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


@registry.register("cosine")
def cosine(threshold: float = 0.95) -> BaseDeduper:
    """Multi-column deduplication using cosine similarity.

    Usage is on multiple columns of a dataframe. Appropriate for numerical
    data.

    Args:
        threshold: the minimum threshold at which similarity between two pairs
            of values will be considered valid for deduplication.

    Returns:
        Instance of `BaseDeduper`.

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
        limiting yourself to using the `cosine` deduplicator with the `Pipeline`
        API using combinations for non null fields.

    Warning:
        Normalization is a standard approach to ensure that the results of
        cosine similarity are valid. Consider [standard
        approaches](https://scikit-learn.org/stable/modules/preprocessing.html#normalization)

    Example:
        Applied to multiple columns:

            import liken as lk

            df = (
                lk.dedupe(df)
                .apply(cosine())
                .drop_duplicates(
                    ("surface are", "ceiling height", "building age", "num_rooms"),
                    keep="first",
                )
                .collect()
            )
    """
    return Cosine(threshold=threshold)


# RULES SUB PKG


@registry.register("isna")
def isna() -> BaseDeduper:
    """Discrete deduper on null/None values.

    Usage is on a single column of a dataframe. Available as the inversion, i.e.
    "not null" using inversion operator: `~isna()`.

    Returns:
        Instance of `BaseDeduper`.

    Example:
        Applied to a single column:

            import liken as lk

            pipeline = lk.rules.pipeline().step(
                [
                    lk.rules.on("email").exact(),
                    ~lk.rules.on("address").isna(),
                ]
            )

            df = (
                lk.dedupe(df)
                .apply(pipeline)
                .drop_duplicates(keep="last")
                .collect()
            )

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


@registry.register("isin")
def isin(values: Iterable) -> BaseDeduper:
    """Discrete deduper for membership testing.

    Usage is on a single column of a dataframe. Available as the inversion, i.e.
    "not in" using inversion operator: `~isin()`.

    Returns:
        Instance of `BaseDeduper`.

    Example:
        Applied to a single column:

            import liken as lk

            pipeline = lk.rules.pipeline().step(
                lk.rules.on("address").isin(values="london")
            )

            df = (
                lk.dedupe(df)
                .apply(pipeline)
                .drop_duplicates(keep="last")
                .collect()
            )

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


@registry.register("str_len")
def str_len(min_len: int = 0, max_len: int | None = None) -> BaseDeduper:
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
        Instance of `BaseDeduper`.

    Example:
        Applied to a single column:

            import liken as lk

            pipeline = lk.rules.pipeline().step(
                [
                    lk.rules.on("email").exact(),
                    lk.rules.on("email").str_len(min_len=10),
                ]
            )

            df = (
                lk.dedupe(df)
                .apply(pipeline)
                .drop_duplicates(keep="last")
                .collect()
            )

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


@registry.register("str_startswith")
def str_startswith(pattern: str, case: bool = True) -> BaseDeduper:
    """Discrete deduper on strings starting with a pattern.

    Usage is on a single column of a dataframe. Available as the inversion, i.e.
    "not starting with pattern" using inversion operator: `~str_startswith()`.

    Deduplication will happen for any pairwise matches that have the same
    `pattern`. Case sensitive unless optionally removed.

    Args:
        pattern: the pattern that the string starts with to be deduplicated
        case: case sensitive, or not.

    Returns:
        Instance of `BaseDeduper`.

    Example:
        Applied to a single column:

            import liken as lk

            pipeline = lk.rules.pipeline().step(
                [
                    lk.rules.on("email").exact(),
                    lk.rules.on("email").str_startswith(pattern="f", case=True),
                ]
            )

            df = (
                lk.dedupe(df)
                .apply(pipeline)
                .drop_duplicates(keep="first")
                .collect()
            )

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


@registry.register("str_endswith")
def str_endswith(pattern: str, case: bool = True) -> BaseDeduper:
    """Discrete deduper on strings ending with a pattern.

    Usage is on a single column of a dataframe. Available as the inversion, i.e.
    "not ending with pattern" using inversion operator: `~str_endswith()`.

    Deduplication will happen for any pairwise matches that have the same
    `pattern`. Case sensitive unless optionally removed.

    Args:
        pattern: the pattern that the string ends with to be deduplicated
        case: case sensitive, or not.

    Returns:
        Instance of `BaseDeduper`.

    Example:
        Applied to a single column:

            import liken as lk

            pipeline = lk.rules.pipeline().step(
                [
                    lk.rules.on("email").exact(),
                    lk.rules.on("email").str_endswith(pattern=".com", case=False),
                ]
            )

            df = (
                lk.dedupe(df)
                .apply(pipeline)
                .drop_duplicates(keep="first")
                .collect()
            )

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


@registry.register("str_contains")
def str_contains(
    pattern: str,
    case: bool = True,
    regex: bool = False,
) -> BaseDeduper:
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
        Instance of `BaseDeduper`.

    Example:
        Applied to a single column:

            import liken as lk

            pipeline = lk.rules.pipeline().step(
                [
                    lk.rules.on("email").exact(),
                    lk.rules.on("email").str_contains(pattern=r"05\\d{3}", regex=True),
                ]
            )

            df = (
                lk.dedupe(df)
                .apply(pipeline)
                .canonicalize(keep="first")
                .collect()
            )

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
