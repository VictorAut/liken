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
        Missing numeric values are filled with 0.0 in that row's vector; the
        column is not dropped. A 0.0 entry contributes nothing to the dot
        product, so a missing value can lower that row's similarity to others
        rather than being ignored.

        If deduplicating columns `col_1`, `col_2` and `col_3` with `cosine`,
        the pairwise similarity is the dot product of the two full rows:

            (`col_1i`, `col_2i`, `col_3i`) . (`col_1j`, `col_2j`, `col_3j`)

        If `col_1i` is missing it is treated as 0.0: it contributes nothing
        to the product and nothing to row `i`'s norm.

        Taking this into account you may find it best to avoid cosine similarity
        calculations for sparse datasets. Alternatively, you may refine your
        approach by either preprocessing the missing values beforehand or by
        limiting yourself to using the `cosine` deduper with the `Pipeline`
        API using combinations for non-null fields.

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
