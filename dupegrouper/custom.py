from typing_extensions import override
import typing

from dupegrouper.strats import ThresholdDedupers, ColumnArrayMixin


# CUSTOM:


class Custom(ThresholdDedupers, ColumnArrayMixin):
    """
    @private
    """

    def __init__(
        self,
        pair_fn: typing.Callable[..., typing.Iterable[tuple[int, int]]],
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
    def _gen_similarity_pairs(self, array) -> typing.Iterator[tuple[int, int]]:
        yield from self._pair_fn(array, **self._kwargs)