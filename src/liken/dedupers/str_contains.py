"""TODO"""

import re
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
class StrContains(
    SingleColumnMixin,
    PredicateDeduper,
):
    """
    Strings contains canonicalizer.

    Defaults to case sensitive. Supports literal substring or regex search.
    """

    _NAME: ClassVar[str] = "str_contains"

    def __init__(self, pattern: str, case: bool = True, regex: bool = False):
        super().__init__(pattern=pattern, case=case, regex=regex)
        self._pattern = pattern
        self._case = case
        self._regex = regex

        if self._regex:
            flags = 0 if self._case else re.IGNORECASE
            self._compiled_pattern = re.compile(self._pattern, flags)

    @override
    def _vectorized_matches(self, array: pa.Array) -> pa.Array:

        if self._regex:
            if self._case:
                return pc.match_substring_regex(array, self._pattern)
            else:
                return pc.match_substring_regex(array, self._pattern, ignore_case=True)

        if self._case:
            return pc.match_substring(array, self._pattern)

        return pc.match_substring(
            pc.utf8_lower(array),
            self._pattern.lower(),
        )

    def __str__(self):
        return self.str_representation(self._NAME)


@dedupers_registry.register("str_contains")
def str_contains(
    pattern: str,
    case: bool = True,
    regex: bool = False,
) -> BaseDeduper:
    """Discrete deduper on general string patterns with regex.

    Usage is on a single column of a dataframe. Available as the inversion, i.e.
    "not containing pattern" using inversion operator: `~str_contains()`.

    Deduplication will happen for any pairwise matches that have the same
    `pattern`. Case sensitive unless optionally removed. Pattern can include
    regex patterns if passed with `regex` arg.

    Args:
        pattern: the pattern that the string ends with to be deduplicated
        case: case sensitive, or not.
        regex: uses regex patterns, or not.

    Returns:
        Instance of `BaseDeduper`.

    Example:
        Applied to a single column:

            import liken as lk

            pipeline = lk.pipeline().step(
                [
                    lk.col("email").exact(),
                    lk.col("email").str_contains(pattern=r"05\\d{3}", regex=True),
                ]
            )

            df = (
                lk.dedupe(df)
                .apply(pipeline)
                .canonicalize(keep="first")
            )

            >>> df
            +------+-----------------------------+
            | id   |           address           |
            +------+-----------------------------+
            |  1   | 12 calle girona, 05891, ES  |
            |  2   |  1A avenida palmas, 05562   |
            |  3   |      901, Spain, 05435      |
            |  4   |     12, santiago, 09945     |
            +------+-----------------------------+

            >>> df # after
            +------+-----------------------------+---------------+
            | id   |           address           |  canonical_id |
            +------+-----------------------------+---------------+
            |  1   | 12 calle girona, 05891, ES  |        1      |
            |  2   |  1A avenida palmas, 05562   |        1      |
            |  3   |      901, Spain, 05435      |        1      |
            |  4   |     12, santiago, 09945     |        4      |
            +------+-----------------------------+---------------+
    """
    return StrContains(pattern=pattern, case=case, regex=regex)
