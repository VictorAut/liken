"""token_overlap deduper"""

from collections.abc import Iterator
from typing import ClassVar
from typing import final

import numpy as np
import pyarrow as pa

from liken.core.deduper import BaseDeduper
from liken.core.deduper import CompoundColumnMixin
from liken.core.deduper import ThresholdDeduper
from liken.core.missing import is_missing
from liken.core.registries import dedupers_registry
from liken.dedupers._blockwise import gram_strip_pairs
from liken.dedupers._blockwise import inverted_index_pairs
from liken.dedupers._blockwise import pick_set_regime
from liken.types import SimilarPairIndices
from liken.validations import require_int


@final
class TokenOverlap(
    CompoundColumnMixin,
    ThresholdDeduper,
):
    """
    Deduplicate sets where such sets contain categorical data.

    None and NaN are both missing: a missing member never matches and
    never counts toward either record's size.

    Pair generation picks between two internal strategies automatically
    per call; no parameter changes.
    """

    _NAME: ClassVar[str] = "token_overlap"

    def __init__(self, threshold: float = 0.95, max_block_memory: int | None = None):
        max_block_memory = require_int("max_block_memory", max_block_memory, low=0, allow_none=True)
        super().__init__(
            threshold=threshold, **({"max_block_memory": max_block_memory} if max_block_memory is not None else {})
        )
        self._threshold = threshold
        self._max_block_memory = max_block_memory

    def _gen_similarity_pairs(self, array: pa.Table) -> Iterator[SimilarPairIndices]:
        columns = [array[col] for col in array.column_names]
        n = array.num_rows

        # None and NaN are both missing, dropped before encoding so they
        # neither match nor count. Distinct members map to dense IDs 0..v-1
        # via one sorted pass.
        row_sets: list[set] = []
        distinct: set = set()
        for i in range(n):
            row_set = {col[i].as_py() for col in columns}
            row_set = {value for value in row_set if not is_missing(value)}
            row_sets.append(row_set)
            distinct |= row_set

        member_ids = {value: value_id for value_id, value in enumerate(sorted(distinct, key=str))}
        members = [
            np.fromiter((member_ids[value] for value in row_set), dtype=np.int64, count=len(row_set))
            for row_set in row_sets
        ]
        sizes = np.fromiter((len(row_set) for row_set in row_sets), dtype=np.int64, count=n)

        if pick_set_regime(members, sizes, self._max_block_memory) == "index":
            yield from inverted_index_pairs(members, sizes, self._threshold, "token_overlap", self._max_block_memory)
        else:
            yield from gram_strip_pairs(members, sizes, self._threshold, "token_overlap", self._max_block_memory)

    def __str__(self):
        return self.str_representation(self._NAME)


@dedupers_registry.register("token_overlap")
def token_overlap(threshold: float = 0.95, max_block_memory: int | None = None) -> BaseDeduper:
    """Multi-column deduplication using set containment.

    Usage is on multiple columns of a dataframe. Appropriate for categorical
    data. Row sets are built exactly as `jaccard` builds them: the set of
    non-missing cell values across the compound columns. There is no
    tokenization. None and NaN are both missing: a missing member is
    excluded from each record's set, so it neither matches values nor adds
    to a record's overlap.

    The containment score is asymmetric: `|A∩B| / min(|A|,|B|)`. A pair
    matches if **either direction** clears the threshold, i.e. the pair
    score is the max of the two directions. This differs from `jaccard`,
    which penalises length asymmetry: a 2-member set fully contained in a
    4-member set scores containment 1.0 but jaccard 0.5.

    An all-missing record's set is empty; its intersection with any other
    set is empty, so an all-missing record matches nothing, including
    another all-missing record (same short-circuit as `jaccard`).

    Args:
        threshold: the minimum threshold at which similarity between two pairs
            of values will be considered valid for deduplication. The
            comparison is inclusive: a pair whose similarity is exactly equal
            to the threshold matches.
        max_block_memory: The approximate memory budget, in bytes, of one
            scored block. Pairs are scored in blocks bounded by this cap;
            smaller blocks lower peak memory at some cost in scoring speed.
            Defaults to a library constant of 64 MiB. A cap too small for
            even one row's strip clamps to a one-row block, so peak memory
            may exceed the cap by that strip.

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
    return TokenOverlap(threshold=threshold, max_block_memory=max_block_memory)
