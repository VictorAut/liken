"""str startswith predicate deduper"""

from typing import ClassVar
from typing import final

import pyarrow as pa
import pyarrow.compute as pc
from typing_extensions import override

from liken.core.deduper import BaseDeduper
from liken.core.deduper import PredicateDeduper
from liken.core.deduper import SingleColumnMixin
from liken.core.registries import dedupers_registry


@final
class StrStartsWith(
    SingleColumnMixin,
    PredicateDeduper,
):
    """
    Strings start with canonicalizer.

    Defaults to case sensitive.

    Regex is not supported, please use `StrContains` otherwise.
    """

    _NAME: ClassVar[str] = "str_startswith"

    def __init__(self, pattern: str, case: bool = True):
        super().__init__(pattern=pattern, case=case)
        self._pattern = pattern
        self._case = case

    @override
    def _vectorized_matches(self, array: pa.Array) -> pa.Array:

        if self._case:
            return pc.starts_with(array, self._pattern)

        return pc.starts_with(
            pc.utf8_lower(array),
            self._pattern.lower(),
        )

    def __str__(self):
        return self.str_representation(self._NAME)


@dedupers_registry.register("str_startswith")
def str_startswith(pattern: str, case: bool = True) -> BaseDeduper:
    """Discrete deduper on strings starting with a pattern.

    Usage is on a single column of a dataframe. Available as the inversion, i.e.
    "not starting with pattern" using inversion operator: `~str_startswith()`.

    Deduplication will happen for any pairwise matches that have the same
    `pattern`. Case sensitive unless optionally removed.

    Args:
        pattern: the pattern that the string starts with to be deduplicated
        case: case sensitive, or not.

    Returns:
        Instance of `BaseDeduper`.

    Example:
        Applied to a single column:

            import liken as lk

            pipeline = lk.pipeline().step(
                [
                    lk.col("email").exact(),
                    lk.col("email").str_startswith(pattern="f", case=True),
                ]
            )

            df = (
                lk.dedupe(df)
                .apply(pipeline)
                .drop_duplicates(keep="first")
            )

            >>> df
            +------+-----------+---------------------+
            | id   |  address  |         email       |
            +------+-----------+---------------------+
            |  1   | new york  |  fizzpop@yahoo.com  |
            |  2   |   london  |  foobar@gmail.co.uk |
            |  3   | marseille |   Flipflop@msn.fr   |
            |  4   |  chicago  |    random@aol.com   |
            +------+-----------+---------------------+

            >>> df # after
            +------+-----------+---------------------+
            | id   |  address  |         email       |
            +------+-----------+---------------------+
            |  1   | new york  |  fizzpop@yahoo.com  |
            |  3   | marseille |   Flipflop@msn.fr   |
            |  4   |  chicago  |    random@aol.com   |
            +------+-----------+---------------------+
    """
    return StrStartsWith(pattern=pattern, case=case)
