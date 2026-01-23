from collections.abc import Iterable, Iterator
from functools import wraps
from typing import Callable, TypeAlias, final

from typing_extensions import override

from dupegrouper._strats_library import ThresholdDedupers
from dupegrouper._types import ArrayLike, SimilarPairIndices


# TYPES:


PairGenerator: TypeAlias = Callable[[ArrayLike], Iterable[SimilarPairIndices]]


# CUSTOM:


@final
class Custom(ThresholdDedupers):
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


def register(f: Callable):
    """TODO"""

    @wraps(f)
    def wrapper(**kwargs):
        return Custom(f, **kwargs)

    return wrapper
