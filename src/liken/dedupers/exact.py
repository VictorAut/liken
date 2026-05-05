"""exact deduper"""

from collections import defaultdict
from typing import ClassVar
from typing import final

import pyarrow as pa
from typing_extensions import override

from liken.core.deduper import BaseDeduper
from liken.core.registries import dedupers_registry


@final
class Exact(BaseDeduper):
    """
    Exact deduper.

    Does not accept a validation mixin (and therefore overrides validation)
    As the exact deduper can be applied to single, or compound columns.
    """

    _NAME: ClassVar[str] = "exact"

    @override
    def validate(self, columns):
        del columns  # Unused
        pass

    @override
    def _gen_similarity_pairs(self, array: pa.Array | pa.Table):
        buckets = defaultdict(list)

        # single column
        if isinstance(array, pa.Array):
            for i, key in enumerate(array):
                buckets[key].append(i)

        # multi column
        if isinstance(array, pa.Table):
            columns = [array[col] for col in array.column_names]

            n = array.num_rows

            for i in range(n):
                key = tuple(col[i].as_py() for col in columns)
                buckets[key].append(i)

        for indices in buckets.values():
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    yield indices[i], indices[j]

    def __str__(self):
        return self.str_representation(self._NAME)


@dedupers_registry.register("exact")
def exact() -> BaseDeduper:
    """Exact Deduplication.

    Can deduplicate a single column, or multiple columns.

    If no dedupers are applied to `Dedupe`, `exact` is applied by default.

    Returns:
        Instance of `BaseDeduper`..

    Example:
        Applied to a single column:

            import liken as lk

            df = (
                lk.dedupe(df)
                .apply(exact())
                .drop_duplicates("address")
            )

        Applied to multiple columns:

            df = (
                lk.dedupe(df)
                .apply(exact())
                .drop_duplicates(("address", "email"))
            )

        E.g.

            >>> df # Before
            +------+-----------+--------------------+
            | id   |  address  |        email       |
            +------+-----------+--------------------+
            |  1   |  london   |  fizzpop@gmail.com |
            |  2   |   null    |  foobar@gmail.com  |
            |  3   |   null    |  foobar@gmail.com  |
            +------+-----------+--------------------+

            >>> df # After
            +------+-----------+---------------------+
            | id   |  address  |        email        |
            +------+-----------+---------------------+
            |  1   |  london   |  fizzpop@gmail.com  |
            |  2   |   null    |  foobar@gmail.com   |
            +------+-----------+---------------------+

        By default `exact` is used when no dedupers are explicitely applied:

            # OK, still dedupes.
            df = Dedupe(df).drop_duplicates("address")
    """
    return Exact()
