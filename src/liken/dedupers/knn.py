"""knn deduper"""

from collections.abc import Iterator
from typing import Any
from typing import ClassVar
from typing import Final
from typing import Literal
from typing import TypeAlias
from typing import cast
from typing import final

import numpy as np
import pyarrow as pa
from scipy.spatial import cKDTree

from liken.core.deduper import BaseDeduper
from liken.core.deduper import CompoundColumnMixin
from liken.core.deduper import ThresholdDeduper
from liken.core.registries import dedupers_registry
from liken.types import SimilarPairIndices
from liken.validations import require_choice
from liken.validations import require_int
from liken.validations import require_type


MetricName: TypeAlias = Literal["euclidean", "cosine"]

_METRIC_NAMES: Final[tuple[str, ...]] = ("euclidean", "cosine")


@final
class Knn(
    CompoundColumnMixin,
    ThresholdDeduper,
):
    """
    Vector nearest-neighbour deduper.
    """

    _NAME: ClassVar[str] = "knn"

    def __init__(self, threshold: float = 0.95, *, k: int, metric: MetricName = "euclidean"):
        k = require_int("k", k, low=1)
        require_type("metric", metric, str)
        metric = cast("MetricName", require_choice("metric", metric, _METRIC_NAMES))
        init_kwargs: dict[str, Any] = {"threshold": threshold, "k": k, "metric": metric}
        super().__init__(**init_kwargs)
        self._threshold = threshold
        self._k = k
        self._metric = metric

    def _gen_similarity_pairs(self, array: pa.Table) -> Iterator[SimilarPairIndices]:

        columns = [array[col].to_numpy(zero_copy_only=False) for col in array.column_names]
        matrix = np.column_stack(columns)

        matrix = np.nan_to_num(matrix, nan=0.0)

        norms = np.linalg.norm(matrix, axis=1)
        zero_norm = norms == 0

        # A zero-norm row (all values missing or all 0.0) matches nothing:
        # kept out of the tree entirely, so it neither emits nor consumes a
        # slot in another row's k best. Under cosine a clamped zero vector
        # would sit at similarity 0.5 from everything; under euclidean at
        # 1/(1+d) from everything nearby.
        present_rows = np.flatnonzero(~zero_norm)
        if present_rows.size < 2:
            return
        data = matrix[present_rows]

        if self._metric == "cosine":
            # cosine distance is not a metric cKDTree can hold; the
            # standard angular pre-normalization makes Euclidean order
            # equal cosine order: on unit vectors, d^2 = 2(1 - cos).
            data = data / np.linalg.norm(data, axis=1)[:, None]

        tree = cKDTree(data)
        m = data.shape[0]

        # one vectorized query for the whole frame; k+1 so self can be
        # dropped without losing a real partner. When k+1 > m the tail
        # is padded with distance inf and index m, dropped below.
        queried: Any = tree.query(data, k=self._k + 1)
        distances, indices = queried

        threshold = self._threshold

        seen: set[tuple[int, int]] = set()

        for pos in range(m):
            i = int(present_rows[pos])
            row_distances = cast("Any", distances[pos])
            row_indices = cast("Any", indices[pos])
            partners = 0
            for distance, j in zip(row_distances, row_indices, strict=True):
                if partners == self._k or j >= m:
                    break
                if j == pos:
                    continue
                partners += 1

                similarity = 1.0 - distance * distance / 2.0 if self._metric == "cosine" else 1.0 / (1.0 + distance)

                if similarity < threshold:
                    continue
                other = int(present_rows[j])
                pair = (min(i, other), max(i, other))
                if pair not in seen:
                    seen.add(pair)
                    yield pair

    def __str__(self):
        return self.str_representation(self._NAME)


@dedupers_registry.register("knn")
def knn(threshold: float = 0.95, *, k: int, metric: MetricName = "euclidean") -> BaseDeduper:
    """Nearest-neighbour deduplication on numeric vectors.

    Usage is on multiple columns of a dataframe, treated as one numeric
    vector per row. Appropriate for numerical data.

    Each row queries its k nearest neighbours; a pair matches when either
    side holds the other in its k best AND the similarity clears the
    threshold. Matching is approximate: a cluster of more than k
    near-identical records may fail to link transitively, because
    union-find still merges transitively but the dropped pairs can split
    it. `k >= n - 1` compares every value against every other.

    Args:
        threshold: the minimum threshold at which similarity between two
            pairs of values will be considered valid for deduplication.
            The comparison is inclusive: a pair whose similarity is
            exactly equal to the threshold matches.
        k: the number of nearest neighbours each row keeps. Required:
            there is no default, because the useful width depends on the
            expected duplicate-cluster size.
        metric: the distance the neighbours are found under.
            - "euclidean" (default): similarity is `1 / (1 + d)`, where
              `d` is the Euclidean distance.
            - "cosine": rows are pre-normalized to unit length and
              neighbours found under Euclidean distance; the similarity
              is `1 - d^2 / 2`, which equals the cosine similarity of
              the original rows up to floating-point error.

    Returns:
        Instance of `BaseDeduper`.

    Note:
        Missing numeric values are filled with 0.0 in that row's vector;
        the column is not dropped. A 0.0 entry contributes nothing to the
        row's norm or its distances. A row that is entirely missing
        (all zeros after the fill) has zero norm: it is normalized to a
        zero vector and matches nothing, not even another all-missing
        row.

        Ties at rank k are resolved by cKDTree's own order. The same
        input order gives the same output, but the tie order is not row
        index order and reordering the column can change the result.

        On a single numeric column `knn` works as the 1-D special case
        and coexists with the `within` deduper: `within` matches on a
        plain absolute tolerance, `knn` adds the top-k restriction.

    Example:
        Applied to multiple columns:

            import liken as lk

            df = (
                lk.dedupe(df)
                .apply(knn(threshold=0.8, k=5))
                .drop_duplicates(
                    ("surface are", "ceiling height", "building age", "num_rooms"),
                    keep="first",
                )
            )
    """
    return Knn(threshold=threshold, k=k, metric=metric)
