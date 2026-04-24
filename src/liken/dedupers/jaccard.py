"""TODO"""

from collections.abc import Iterator
from typing import ClassVar
from typing import final

import pyarrow as pa

from liken.core.deduper import BaseDeduper
from liken.core.deduper import CompoundColumnMixin
from liken.core.deduper import ThresholdDeduper
from liken.core.registries import dedupers_registry
from liken.types import SimilarPairIndices


@final
class Jaccard(
    CompoundColumnMixin,
    ThresholdDeduper,
):
    """
    Deduplicate sets where such sets contain categorical data.
    """

    _NAME: ClassVar[str] = "jaccard"

    def _gen_similarity_pairs(self, array: pa.Table) -> Iterator[SimilarPairIndices]:
        columns = [array[col] for col in array.column_names]
        n = array.num_rows

        # Build row sets directly
        sets = [{col[i].as_py() for col in columns if col[i].as_py() is not None} for i in range(n)]

        for idx in range(n):
            for idy in range(idx + 1, n):
                intersection = sets[idx] & sets[idy]

                if not intersection:
                    continue  # no match

                union = sets[idx] | sets[idy]

                if not union:
                    continue  # zero div: guardrail

                if len(intersection) / len(union) > self._threshold:
                    yield idx, idy

    def __str__(self):
        return self.str_representation(self._NAME)


@dedupers_registry.register("jaccard")
def jaccard(threshold: float = 0.95) -> BaseDeduper:
    """Multi-column deduplication using jaccard similarity.

    Usage is on multiple columns of a dataframe. Appropriate for categorical
    data. Null types are handled out-of-box with jaccard, they are simply
    considered another category of a given field.

    Args:
        threshold: the minimum threshold at which similarity between two pairs
            of values will be considered valid for deduplication.

    Returns:
        Instance of `BaseDeduper`.

    Example:
        Applied to multiple columns:

            import liken as lk

            df = (
                lk.dedupe(df)
                .apply(jaccard())
                .drop_duplicates(
                    ("account", "status", "country", "property"),
                    keep="first",
                )
            )

        E.g.

            >>> df
            +------+-----------+----------+----------+-----------+
            | id   |  account  |  status  |  country |  property |
            +------+-----------+----------+----------+-----------+
            |  1   |  reddit   |  married |    UK    |  house    |
            |  2   |  flickr   |  married |    UK    |  house    |
            |  3   | pinterest |  single  |  Germany |  flat     |
            +------+-----------+----------+----------+-----------+

            >>> df # After deduplication
            +------+-----------+----------+----------+-----------+
            | id   |  account  |  status  |  country |  property |
            +------+-----------+----------+----------+-----------+
            |  1   |  reddit   |  married |    UK    |  house    |
            |  3   | pinterest |  single  |  Germany |  flat     |
            +------+-----------+----------+----------+-----------+
    """
    return Jaccard(threshold=threshold)
