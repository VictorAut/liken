"""cosine deduper"""

from collections.abc import Iterator
from typing import ClassVar
from typing import final

import numpy as np
import pyarrow as pa
from typing_extensions import override

from liken.core.deduper import BaseDeduper
from liken.core.deduper import CompoundColumnMixin
from liken.core.deduper import ThresholdDeduper
from liken.core.registries import dedupers_registry
from liken.types import SimilarPairIndices


@final
class Cosine(
    CompoundColumnMixin,
    ThresholdDeduper,
):
    """
    Deduplicate sets where such sets contain numeric data.
    """

    _NAME: ClassVar[str] = "cosine"

    @override
    def _gen_similarity_pairs(self, array: pa.Table) -> Iterator[SimilarPairIndices]:

        columns = [array[col].to_numpy(zero_copy_only=False) for col in array.column_names]
        matrix = np.column_stack(columns)

        matrix = np.nan_to_num(matrix, nan=0.0)

        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1

        normalized = matrix / norms[:, None]

        n = normalized.shape[0]

        for i in range(n):
            sims = normalized[i] @ normalized[i + 1 :].T

            for offset, val in enumerate(sims):
                if val > self._threshold:
                    yield i, i + 1 + offset

    def __str__(self):
        return self.str_representation(self._NAME)


@dedupers_registry.register("cosine")
def cosine(threshold: float = 0.95) -> BaseDeduper:
    """Multi-column deduplication using cosine similarity.

    Usage is on multiple columns of a dataframe. Appropriate for numerical
    data.

    Args:
        threshold: the minimum threshold at which similarity between two pairs
            of values will be considered valid for deduplication.

    Returns:
        Instance of `BaseDeduper`.

    Note:
        In the case of null types, that column is ignore, and only the
        similarity is taken of the remaining columns is taken.

        So, if deduplicating columns `col_1`, `col_2` and `col_3` with `cosine`,
        any similarity is usually the dot product for a given pairwise evaluation
        i.e.

            (`col_1i`, `col_2i`, `col_3i`) . (`col_1j`, `col_2j`, `col_3j`)

        However, if `col_1i` is Null then the following is evaluated:

            (`col_2i`, `col_3i`) . (`col_2j`, `col_3j`)

        Additionally, if `col_j2` is *also* Null then the following is evaluated:

            (`col_3i`) . (`col_3j`)

        Taking this into account you may find it best to avoid cosine similarity
        calculations for sparse datasets. Alternatively, you may opt to your
        approach by either preprocessing the Nulls beforehand, or, by
        limiting yourself to using the `cosine` deduplicator with the `Pipeline`
        API using combinations for non null fields.

    Warning:
        Normalization is a standard approach to ensure that the results of
        cosine similarity are valid. Consider [standard
        approaches](https://scikit-learn.org/stable/modules/preprocessing.html#normalization)

    Example:
        Applied to multiple columns:

            import liken as lk

            df = (
                lk.dedupe(df)
                .apply(cosine())
                .drop_duplicates(
                    ("surface are", "ceiling height", "building age", "num_rooms"),
                    keep="first",
                )
            )
    """
    return Cosine(threshold=threshold)
