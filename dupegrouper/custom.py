from collections.abc import Iterable, Iterator
from typing_extensions import override
from typing import Callable

from dupegrouper.strats import ThresholdDedupers, ColumnArrayMixin
from dupegrouper.types import ArrayLike, SimilarPairIndices


# CUSTOM:


class Custom(ThresholdDedupers, ColumnArrayMixin):
    """
    @private
    """

    def __init__(
        self,
        pair_fn: Callable[[ArrayLike], Iterable[SimilarPairIndices]],
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
    def wrapper(**kwargs):
        return Custom(f, **kwargs)
    return wrapper