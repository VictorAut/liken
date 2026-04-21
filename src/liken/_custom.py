"""Define custom dedupers"""

from collections.abc import Iterable
from collections.abc import Iterator
from typing import Callable
from typing import TypeAlias
from typing import final

from typing_extensions import override

from liken._dedupers import ThresholdDeduper
from liken._types import SimilarPairIndices


# TYPES:


PairGenerator: TypeAlias = Callable[[Iterable], Iterable[SimilarPairIndices]]


# CUSTOM:


@final
class Custom(ThresholdDeduper):
    """
    Inherits from Threshold Dedupers for a generalised approach.

    Overrides _gen_similarity_pairs to accept a custom callable, which albeit
    this class being derived from the ThresholdDeduper class, can nevertheless
    be implemented such that it produces predicate results.
    """

    def __init__(
        self,
        pair_fn: PairGenerator,
        /,
        *args,
        **kwargs,
    ):
        super().__init__(
            pair_fn=pair_fn,
            *args,
            **kwargs,
        )
        self._pair_fn = pair_fn
        self._args = args
        self._kwargs = kwargs

    @override
    def validate(self, columns):
        """No validation such that custom can be applied to single or
        compound column
        """
        del columns  # Unused
        pass

    @override
    def _gen_similarity_pairs(self, array) -> Iterator[SimilarPairIndices]:
        """generator or function implementation"""
        array: list = array.to_pylist()
        yield from self._pair_fn(array, *self._args, **self._kwargs)

    def __str__(self):
        return self.__repr__()
