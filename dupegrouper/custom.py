from collections.abc import Iterable, Iterator
from functools import wraps
from typing_extensions import override
from typing import Callable, TypeAlias

from dupegrouper.strats import ThresholdDedupers, ColumnArrayMixin
from dupegrouper.types import ArrayLike, SimilarPairIndices


PairGenerator: TypeAlias = Callable[[ArrayLike], Iterable[SimilarPairIndices]]


# CUSTOM:


class Custom(ThresholdDedupers, ColumnArrayMixin):
    """
    @private
    """

    def __init__(
        self,
        pair_fn: PairGenerator,
        /,
        **kwargs,
    ):
        self._pair_fn = pair_fn
        self._kwargs = kwargs

    @override  # As no validation mixin provided
    def validate(self, columns):
        del columns  # Unused
        pass

    @override
    def _gen_similarity_pairs(self, array) -> Iterator[SimilarPairIndices]:
        yield from self._pair_fn(array, **self._kwargs)


# REGISTER:


def register(f: Callable):
    # @wraps(f)
    def wrapper(**kwargs):
        return Custom(f, **kwargs)
    return wrapper