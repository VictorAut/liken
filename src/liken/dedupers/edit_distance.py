"""edit_distance deduper"""

from collections.abc import Iterator
from typing import ClassVar
from typing import final

import pyarrow as pa
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from liken.core.deduper import BaseDeduper
from liken.core.deduper import SingleColumnMixin
from liken.core.missing import partition_missing
from liken.core.registries import dedupers_registry
from liken.types import SimilarPairIndices


@final
class EditDistance(
    SingleColumnMixin,
    BaseDeduper,
):
    """
    Edit distance deduper

    Two values are matched when their Levenshtein distance is at most
    `max_distance`. The bound is absolute: it does not scale with string
    length, unlike the ratio scorers of `fuzzy`.

    Missing values match only each other, never a value, regardless of
    distance.
    """

    _NAME: ClassVar[str] = "edit_distance"

    def __init__(self, max_distance: int = 2):
        super().__init__(max_distance=max_distance)
        self._max_distance = max_distance

        if not isinstance(max_distance, int) or max_distance < 0:
            raise ValueError("The max_distance value must be a non-negative integer")

    def _gen_similarity_pairs(self, array: pa.Array) -> Iterator[SimilarPairIndices]:
        values: list = array.to_pylist()

        missing, present = partition_missing(values)

        for i in missing[1:]:
            yield missing[0], i

        if not present:
            return

        # the scorer receives non-missing values only
        scored = [values[i] for i in present]

        for pos, s1 in enumerate(scored):
            if pos + 1 >= len(scored):
                break

            distances = process.cdist(
                [s1],
                scored[pos + 1 :],
                scorer=Levenshtein.distance,
            )[0]

            for offset, distance in enumerate(distances):
                if distance > self._max_distance:
                    continue

                yield present[pos], present[pos + 1 + offset]

    def __str__(self):
        return self.str_representation(self._NAME)


@dedupers_registry.register("edit_distance")
def edit_distance(max_distance: int = 2) -> BaseDeduper:
    """Edit distance deduplication.

    Usage is on single columns of a dataframe.

    Two values are matched when their Levenshtein distance - the number of
    single-character edits between them - is at most `max_distance`. The
    bound is inclusive and absolute: it does not scale with string length.
    Use it for short codes, such as postcodes, phone numbers and product
    codes, where a relative similarity threshold exists otherwise.

    Missing values are matched only against other missing values, never
    against other values.

    Args:
        max_distance: The maximum Levenshtein distance at which two values
            are considered valid for deduplication. Defaults to 2.

    Returns:
        Instance of `BaseDeduper`.

    Example:
        Applied to a single column:

            import liken as lk

            df = (
                lk.dedupe(df)
                .apply({"postcode": edit_distance(max_distance=2)})
                .drop_duplicates()
            )

        E.g.

            >>> df # Before
            +------+-----------+----------------------+
            | id   | postcode  |         email        |
            +------+-----------+----------------------+
            |  1   |  OL5 9PL  |  fizzpop@gmail.com   |
            |  2   |   null    |  foobar@gmail.com    |
            |  3   |  OL5 9P1  |  foobar@gmail.co.uk  |
            +------+-----------+----------------------+

            >>> df # After
            +------+-----------+----------------------+
            | id   | postcode  |         email        |
            +------+-----------+----------------------+
            |  1   |  OL5 9PL  |  fizzpop@gmail.com   |
            |  3   |  OL5 9P1  |  foobar@gmail.co.uk  |
            +------+-----------+----------------------+
    """
    return EditDistance(max_distance=max_distance)
