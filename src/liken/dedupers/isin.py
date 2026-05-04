"""isin predicate deduper"""

from typing import ClassVar
from typing import Iterable
from typing import final

from typing_extensions import override

from liken.core.deduper import BaseDeduper
from liken.core.deduper import PredicateDeduper
from liken.core.deduper import SingleColumnMixin
from liken.core.registries import dedupers_registry


@final
class IsIn(
    SingleColumnMixin,
    PredicateDeduper,
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


@dedupers_registry.register("isin")
def isin(values: Iterable) -> BaseDeduper:
    """Discrete deduper for membership testing.

    Usage is on a single column of a dataframe. Available as the inversion, i.e.
    "not in" using inversion operator: `~isin()`.

    Returns:
        Instance of `BaseDeduper`.

    Example:
        Applied to a single column:

            import liken as lk

            pipeline = lk.pipeline().step(
                lk.col("address").isin(values="london")
            )

            df = (
                lk.dedupe(df)
                .apply(pipeline)
                .drop_duplicates(keep="last")
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
