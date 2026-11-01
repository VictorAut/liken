"""edit_distance deduper"""

from collections.abc import Callable
from collections.abc import Iterator
from typing import Any
from typing import ClassVar
from typing import Final
from typing import Literal
from typing import TypeAlias
from typing import cast
from typing import final

import pyarrow as pa
from rapidfuzz.distance import DamerauLevenshtein
from rapidfuzz.distance import Hamming
from rapidfuzz.distance import Indel
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.distance import Levenshtein

from liken.core.deduper import BaseDeduper
from liken.core.deduper import SingleColumnMixin
from liken.core.missing import partition_missing
from liken.core.registries import dedupers_registry
from liken.dedupers._blockwise import DEFAULT_BLOCK_MEMORY
from liken.dedupers._blockwise import block_rows_for
from liken.dedupers._blockwise import probe_itemsize
from liken.dedupers._blockwise import resolve_workers
from liken.dedupers._blockwise import similar_pairs_blockwise
from liken.exceptions import InvalidArgumentValueError
from liken.types import SimilarPairIndices
from liken.validations import require_choice
from liken.validations import require_int
from liken.validations import require_real_number
from liken.validations import require_type


MetricName: TypeAlias = Literal["levenshtein", "damerau_levenshtein", "jaro_winkler", "indel", "hamming"]

# The four non-levenshtein metrics express `max_distance` as a maximum
# normalized distance in [0, 1]. `process.cdist` outputs float32 by
# default, whose rounding error (measured up to ~2.4e-8) exceeds the
# tolerance below, so those metrics run with `dtype="float64"`. The guard
# then only admits scores equal to the boundary at any meaningful
# precision, mirroring `fuzzy`'s tolerance but widening admission instead:
# a distance at or below the boundary matches, where `fuzzy`'s similarity
# must reach the boundary.
_BOUNDARY_TOLERANCE: Final[float] = 1e-9

_METRICS: Final[dict[str, Callable]] = {
    "levenshtein": Levenshtein.distance,
    "damerau_levenshtein": DamerauLevenshtein.normalized_distance,
    "jaro_winkler": JaroWinkler.normalized_distance,
    "indel": Indel.normalized_distance,
    "hamming": Hamming.normalized_distance,
}

_METRIC_NAMES: Final[tuple[str, ...]] = tuple(_METRICS)

# Levenshtein keeps an absolute edit-count bound; the other four metrics
# take a maximum normalized distance and need no default carried over from
# the absolute contract.
_DEFAULT_MAX_DISTANCE: Final[int] = 2


@final
class EditDistance(
    SingleColumnMixin,
    BaseDeduper,
):
    """
    Edit distance deduper

    Two values are matched when their distance is at most `max_distance`.
    The bound is inclusive. With the default `metric` it is an absolute
    edit count that does not scale with string length, unlike the ratio
    scorers of `fuzzy`.

    Missing values match only each other, never a value, regardless of
    distance.
    """

    _NAME: ClassVar[str] = "edit_distance"

    def __init__(
        self,
        max_distance: int | float = 2,
        metric: MetricName = "levenshtein",
        workers: int | None = None,
        max_block_memory: int | None = None,
    ):
        require_type("metric", metric, str)
        metric = cast("MetricName", require_choice("metric", metric, _METRIC_NAMES))
        if metric == "levenshtein":
            max_distance = require_int("max_distance", max_distance, low=0)
        else:
            max_distance = require_real_number("max_distance", max_distance, low=0, high=1, high_inclusive=True)
        workers = require_int("workers", workers, low=-1, allow_none=True)
        if workers == 0:
            raise InvalidArgumentValueError("`workers` must be a nonzero integer or -1, got 0")
        max_block_memory = require_int("max_block_memory", max_block_memory, low=0, allow_none=True)
        init_kwargs: dict[str, Any] = {"max_distance": max_distance, "metric": metric}
        if workers is not None:
            init_kwargs["workers"] = workers
        if max_block_memory is not None:
            init_kwargs["max_block_memory"] = max_block_memory
        super().__init__(**init_kwargs)
        self._max_distance = max_distance
        self._metric = metric
        self._workers = workers
        self._max_block_memory = max_block_memory

    def _gen_similarity_pairs(self, array: pa.Array) -> Iterator[SimilarPairIndices]:
        values: list = array.to_pylist()

        missing, present = partition_missing(values)

        for i in missing[1:]:
            yield missing[0], i

        if not present:
            return

        # the scorer receives non-missing values only
        scored = [values[i] for i in present]

        scorer = _METRICS[self._metric]

        # levenshtein keeps its absolute integer contract on cdist's default
        # int dtype; the normalized metrics run on float64 so the boundary
        # tolerance stays meaningful against float32 rounding.
        dtype = "float64" if self._metric != "levenshtein" else None
        bound = cast("float", self._max_distance)
        if dtype is not None:
            bound += _BOUNDARY_TOLERANCE
        else:
            bound = float(bound)

        max_block_memory = DEFAULT_BLOCK_MEMORY if self._max_block_memory is None else int(self._max_block_memory)
        block_rows = block_rows_for(len(scored), probe_itemsize(scorer, dtype=dtype), max_block_memory)
        workers = resolve_workers(self._workers, distributed=self.distributed)

        for r, c in similar_pairs_blockwise(
            scored,
            compare=scorer,
            low=True,
            bound=bound,
            block_rows=block_rows,
            workers=workers,
            dtype=dtype,
        ):
            yield present[r], present[c]

    def __str__(self):
        return self.str_representation(self._NAME)


@dedupers_registry.register("edit_distance")
def edit_distance(
    max_distance: int | float = _DEFAULT_MAX_DISTANCE,
    metric: MetricName = "levenshtein",
    workers: int | None = None,
    max_block_memory: int | None = None,
) -> BaseDeduper:
    """Edit distance deduplication.

    Usage is on single columns of a dataframe.

    Two values are matched when their distance - the metric-dependent
    edit cost between them - is at most `max_distance`. The bound is
    inclusive. With the default metric it is an absolute edit count that
    does not scale with string length. Use it for short codes, such as
    postcodes, phone numbers and product codes, where a relative
    similarity threshold exists otherwise.

    Missing values are matched only against other missing values, never
    against other values.

    Args:
        max_distance: The maximum distance at which two values are
            considered valid for deduplication. With `levenshtein` it is
            an absolute edit count, defaulting to 2. With every other
            metric it is a maximum normalized distance in [0, 1], where 0
            matches identical values only and 1 matches everything; it
            must then be passed explicitly, since the default 2 is out of
            range. Defaults to 2.
        metric: The distance metric. Defaults to "levenshtein". Options
            are
            - "levenshtein": the number of single-character edits, as an
              absolute count.
            - "damerau_levenshtein": like levenshtein, but transpositions
              count as one edit, as a maximum normalized distance.
            - "jaro_winkler": a similarity weighted towards shared
              prefixes, scaled to a maximum normalized distance. Short
              strings such as names, postcodes and product codes score
              closer than edit counts alone suggest.
            - "indel": insertions and deletions only, substitutions count
              as two edits, as a maximum normalized distance.
            - "hamming": substitutions between equal-length positions,
              padded when one value is longer, as a maximum normalized
              distance over the longer length.

            Lower distances are closer, the opposite direction of
            `fuzzy`'s `threshold`.
        workers: The number of worker threads used for scoring. This
            parameter has multiple effective defaults: `None` chooses
            automatically, using all CPU cores locally or for single node
            backends, but a single worker under a distributed backend.`-1`
            overrides the default and uses all cores, else choose an integer
            number for the desired number of cores.
        max_block_memory: The approximate memory budget, in bytes, of one
            scored block. Values are scored in blocks bounded by this
            cap; smaller blocks lower peak memory at some cost in scoring
            speed. Defaults to a library constant of 64 MiB. A cap too
            small for even one row's score band clamps to a one-row
            block, so peak memory may exceed the cap by that band.

    Returns:
        Instance of `BaseDeduper`.

    Raises:
        InvalidArgumentTypeError: if `max_distance` is not an integer for
            `levenshtein`, not a real number for the other metrics, or if
            `metric` is not a string.
        InvalidArgumentValueError: if `max_distance` is negative, or, for
            the normalized metrics, outside [0, 1], or if `metric` is not
            one of the options.

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
    return EditDistance(max_distance=max_distance, metric=metric, workers=workers, max_block_memory=max_block_memory)
