"""isna predicate deduper"""

from typing import ClassVar
from typing import final

import pyarrow as pa
from typing_extensions import override

from liken.core.deduper import BaseDeduper
from liken.core.deduper import PredicateDeduper
from liken.core.deduper import SingleColumnMixin
from liken.core.missing import is_missing
from liken.core.registries import dedupers_registry


@final
class IsNA(
    SingleColumnMixin,
    PredicateDeduper,
):
    """
    Deduplicates all missing / null values into a single group.

    Inversion operator here calls it's own negation class. The paired
    `_gen_similarity_pairs` implementations are deliberately not factored
    through `_NegatedPredicateDeduper` because NA needs its own inversion
    logic.
    """

    _NAME: ClassVar[str] = "isna"

    @override
    def _gen_similarity_pairs(self, array: pa.Array):
        values: list = array.to_pylist()

        missing: list[int] = [i for i, value in enumerate(values) if is_missing(value)]

        for i in range(len(missing)):
            for j in range(i + 1, len(missing)):
                yield missing[i], missing[j]

    def __str__(self):
        return self.str_representation(self._NAME)

    def __invert__(self):
        return _NotNA()


@final
class _NotNA(
    SingleColumnMixin,
    PredicateDeduper,
):
    """
    Deduplicate all non-NA / non-null values.

    "not a match" for not null does not hold like it does for other predicate
    Dedupers.
    """

    _NAME: ClassVar[str] = "~isna"

    @override
    def _gen_similarity_pairs(self, array: pa.Array):
        values: list = array.to_pylist()

        present: list[int] = [i for i, value in enumerate(values) if not is_missing(value)]

        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                yield present[i], present[j]

    def __str__(self):
        return self.str_representation(self._NAME)


@dedupers_registry.register("isna")
def isna() -> BaseDeduper:
    """Discrete deduper on null/None values.

    Usage is on a single column of a dataframe. Available as the inversion, i.e.
    "not null" using inversion operator: `~isna()`.

    Returns:
        Instance of `BaseDeduper`.

    Example:
        Applied to a single column:

            import liken as lk

            pipeline = lk.pipeline().step(
                [
                    lk.col("email").exact(),
                    ~lk.col("address").isna(),
                ]
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
