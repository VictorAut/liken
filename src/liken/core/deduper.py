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

from collections.abc import Iterator
from typing import TYPE_CHECKING
from typing import Protocol
from typing import Self

import pyarrow as pa
import pyarrow.compute as pc
from networkx.utils.union_find import UnionFind
from typing_extensions import override

from liken.constants import CANONICAL_ID
from liken.preprocessors import Preprocessor


if TYPE_CHECKING:
    from liken.core.wrapper import DF
    from liken.types import Columns
    from liken.types import Keep
    from liken.types import MultiComponents
    from liken.types import SimilarPairIndices
    from liken.types import SingleComponents


# INTERFACE:


class Base(Protocol):
    wdf: DF
    with_na_placeholder: bool

    def set_frame(self, wdf: DF) -> Self: ...
    def _gen_similarity_pairs(self, array: pa.Array | pa.Table) -> Iterator[SimilarPairIndices]: ...
    def build_union_find(
        self,
        columns: Columns,
        preprocessors: list[Preprocessor],
        predicate: set = set(),
    ) -> tuple[UnionFind[int], int]: ...
    def canonicalizer(
        self,
        *,
        components: SingleComponents | MultiComponents,
        drop_duplicates: bool,
        keep: Keep,
    ) -> DF: ...
    def str_representation(self, name: str) -> str: ...
    def validate(self, columns: Columns) -> None: ...

    @staticmethod
    def preprocess(array: pa.Array | pa.Table, preprocessors: list[Preprocessor]) -> pa.Array | pa.Table: ...


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

    def set_frame(self, wdf: DF) -> Self:
        """Inject dataframe and interface methods"""
        self.wdf: DF = wdf
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
    ) -> DF:
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


# PREDICATE DEDUPERS:


class PredicateDeduper(BaseDeduper):
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

            if indices:
                root = indices[0]
                for i in indices[1:]:
                    yield root, i
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


class _NegatedPredicateDeduper(PredicateDeduper):
    """
    Composable deduplication instance that inverts the results of any predicate
    deduper (except IsNA deduper which follows it's own inversion logic).
    """

    def __init__(self, inner: PredicateDeduper):
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


# THRESHOLD DEDUPERS:


class ThresholdDeduper(BaseDeduper):
    """
    Base instance of dedupers that implement any similarity comparison
    mechanism.
    """

    def __init__(self, threshold: float = 0.95, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self._threshold = threshold

        if not (0 <= threshold < 1):
            raise ValueError("The threshold value must be greater or equal to 0 and less than 1")
