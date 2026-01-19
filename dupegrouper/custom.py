from collections.abc import Iterable, Iterator
from functools import wraps
from typing import Callable, TypeAlias, final

from typing_extensions import override

from dupegrouper.strats_library import ColumnArrayMixin, ThresholdDedupers
from dupegrouper.types import ArrayLike, SimilarPairIndices

PairGenerator: TypeAlias = Callable[[ArrayLike], Iterable[SimilarPairIndices]]


# CUSTOM:


@final
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
        super().__init__(
            pair_fn=pair_fn,
            **kwargs,
        )
        self._pair_fn = pair_fn
        self._kwargs = kwargs

    def validate(self, columns):
        del columns  # Unused
        pass

    @override
    def _gen_similarity_pairs(self, array) -> Iterator[SimilarPairIndices]:
        yield from self._pair_fn(array, **self._kwargs)

    def __str__(self):
        return self.__repr__()


# REGISTER:


def register(f: Callable):
    @wraps(f)
    def wrapper(**kwargs):
        return Custom(f, **kwargs)

    return wrapper
