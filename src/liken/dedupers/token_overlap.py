"""token_overlap deduper"""

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
class TokenOverlap(
    CompoundColumnMixin,
    ThresholdDeduper,
):
    """
    Deduplicate sets where such sets contain categorical data.
    """

    _NAME: ClassVar[str] = "token_overlap"

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

                denominator = min(len(sets[idx]), len(sets[idy]))

                if len(intersection) / denominator >= self._threshold:
                    yield idx, idy

    def __str__(self):
        return self.str_representation(self._NAME)


@dedupers_registry.register("token_overlap")
def token_overlap(threshold: float = 0.95) -> BaseDeduper:
    """Multi-column deduplication using set containment.

    Usage is on multiple columns of a dataframe. Appropriate for categorical
    data. Row sets are built exactly as `jaccard` builds them: the set of
    non-null cell values across the compound columns. There is no
    tokenization.

    The containment score is asymmetric: `|A∩B| / min(|A|,|B|)`. A pair
    matches if **either direction** clears the threshold, i.e. the pair
    score is the max of the two directions. This differs from `jaccard`,
    which penalises length asymmetry: a 2-member set fully contained in a
    4-member set scores containment 1.0 but jaccard 0.5.

    Null handling out-of-box: `None` members are excluded from each
    record's set, so they neither match values nor add to a record's
    containment. A float NaN member stays in the set where the backend
    keeps it. An all-null record's set is empty; its intersection with any
    other set is empty, so an all-null record matches nothing, including
    another all-null record (same short-circuit as `jaccard`).

    Args:
        threshold: the minimum threshold at which similarity between two pairs
            of values will be considered valid for deduplication. The
            comparison is inclusive: a pair whose similarity is exactly equal
            to the threshold matches.

    Returns:
        Instance of `BaseDeduper`.

    Example:
        Applied to multiple columns:

            import liken as lk

            df = (
                lk.dedupe(df)
                .apply(token_overlap())
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
            |  2   |  reddit   |  married |    UK    |           |
            |  3   | pinterest |  single  |  Germany |  flat     |
            +------+-----------+----------+----------+-----------+

            >>> df # After deduplication
            +------+-----------+----------+----------+-----------+
            | id   |  account  |  status  |  country |  property |
            +------+-----------+----------+----------+-----------+
            |  1   |  reddit   |  married |    UK    |  house    |
            |  3   | pinterest |  single  |  Germany |  flat     |
            +------+-----------+----------+----------+-----------+

        Row 1's set {reddit, married, UK} is fully contained in row 0's
        {reddit, married, UK, house}: containment 1.0 against jaccard 0.75.
    """
    return TokenOverlap(threshold=threshold)
