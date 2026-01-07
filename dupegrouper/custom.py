from collections.abc import Iterable, Iterator
from typing_extensions import override
from typing import Callable, TypeAlias

from dupegrouper.strats import ThresholdDedupers, ColumnArrayMixin
from dupegrouper.types import ArrayLike, SimilarPairIndices


PairGenerator: TypeAlias = Callable[[ArrayLike], Iterable[SimilarPairIndices]]


# REGISTRY:


plugin_registry: dict[str, PairGenerator] = {}


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
    def wrapper(**kwargs):
        plugin_registry[f.__name__] = f
        return Custom(f, **kwargs)
    return wrapper

def get_plugins():
    result = {}
    for name, fn in plugin_registry.items():
        result[name] = (lambda _fn=fn, **kwargs: Custom(_fn, **kwargs))
    return result