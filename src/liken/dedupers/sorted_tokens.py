"""sorted tokens predicate deduper"""

from collections.abc import Iterator
from typing import ClassVar
from typing import cast
from typing import final

import pyarrow as pa
from typing_extensions import override

from liken.core.deduper import BaseDeduper
from liken.core.deduper import PredicateDeduper
from liken.core.deduper import SingleColumnMixin
from liken.core.missing import is_missing
from liken.core.missing import partition_missing
from liken.core.registries import dedupers_registry
from liken.exceptions import InvalidArgumentTypeError
from liken.types import SimilarPairIndices


@final
class SortedTokens(
    SingleColumnMixin,
    PredicateDeduper,
):
    """
    Deduplicates values built from the same words in any order.

    Each value is tokenised on whitespace, the tokens are sorted, and
    values whose sorted token sequences are equal are deduplicated.
    Matching is word-order only: it does not cover typos, character
    transpositions, or case differences — values are used raw.

    Inversion follows the per-row predicate contract: `_matches` is "the
    value is a non-missing string with at least one token", so
    `~sorted_tokens` groups zero-token values, and a missing value matches
    nothing on either polarity.
    """

    _NAME: ClassVar[str] = "sorted_tokens"

    @override
    def _matches(self, value: object) -> bool:
        """Per-row predicate for the negated path: a non-missing string
        with at least one token."""
        if is_missing(value):
            return False
        # The negated path has passed `_vectorized_matches`, so the column
        # is string-typed and the value is a Python string.
        return bool(cast("str", value).split())

    def _require_string_array(self, array: pa.Array) -> None:
        """Raise unless `array` is a string array, or a `null`-typed array
        (e.g. the all-empty column: nothing to tokenise)."""
        # A `null`-typed array has no strings to check, so the string
        # check passes it through; a non-empty all-null array then yields
        # only the missing-star pairs.
        if (
            array.type != pa.null()
            #
            and not (pa.types.is_string(array.type) or pa.types.is_large_string(array.type))
        ):
            raise InvalidArgumentTypeError(f"`sorted_tokens` requires a string column, got Arrow type '{array.type}'")

    @override
    def _vectorized_matches(self, array: pa.Array) -> pa.Array | None:
        # Type check only: the raise propagates on the negated path, where
        # a non-string column would otherwise silently star-group.
        self._require_string_array(array)
        return None

    @override
    def _gen_similarity_pairs(self, array: pa.Array) -> Iterator[SimilarPairIndices]:
        self._require_string_array(array)

        values: list = array.to_pylist()

        missing, present = partition_missing(values)

        # Missing values match only each other, star-shaped from the first
        # missing index.
        for i in missing[1:]:
            yield missing[0], i

        if not present:
            return

        # Bucket on the sorted token sequence. `str.split()` treats runs of
        # any whitespace as separators and ignores padding, so values that
        # differ only in spacing share a bucket. Sorting on the token
        # strings is code-point order, no locale collation, so the key is
        # deterministic across backends. A zero-token value produces the
        # empty key; the `present` filter below keeps it unmatched — there
        # are no words to swap.
        buckets: dict[tuple[str, ...], list[int]] = {}
        for pos in present:
            key = tuple(sorted(values[pos].split()))
            if key:
                buckets.setdefault(key, []).append(pos)

        for indices in buckets.values():
            # Star-shaped from the bucket's first index (predicate
            # convention). Bucketing is O(n); no all-pairs scan.
            first = indices[0]
            for pos in indices[1:]:
                a, b = (first, pos) if first < pos else (pos, first)
                yield a, b

    def __str__(self):
        return self.str_representation(self._NAME)


@dedupers_registry.register("sorted_tokens")
def sorted_tokens() -> BaseDeduper:
    """Word-order predicate deduplication.

    Usage is on a single string column of a dataframe. Available as the
    inversion using the inversion operator: `~sorted_tokens()`.

    Each value is tokenised on whitespace and the tokens are sorted; two
    values match when their sorted token sequences are equal. "John Smith"
    and "Smith John" match. Runs of whitespace are one separator and
    padding is ignored: `"  John   Smith  "` matches `"John Smith"`.

    Matching is word-order only, and values are used raw:

    - Typos are out of scope — use `fuzzy` or `edit_distance`.
    - Character transpositions ("A1324" vs "A1234") are out of scope — the
      swap unit is the whole token, not the character.
    - Case folding is not applied — "John smith" and "john Smith" do not
      match.

    The token boundary is whitespace only: punctuation placement is part
    of the token, so "Main St." and "St. Main" do not match.

    A value with no tokens (an empty or whitespace-only string) matches
    nothing on the positive path. Under `~sorted_tokens` the groups
    invert: zero-token values group together, values with tokens do not
    match, and a missing value still matches nothing. Negation star-groups
    every zero-token value into one group through all-pairs emission, so
    the work is quadratic in the number of zero-token values; keep
    zero-token-heavy columns out of `~sorted_tokens`.

    Returns:
        Instance of `BaseDeduper`.

    Raises:
        InvalidArgumentTypeError: if the column is not a string column at
            execution.

    Example:
        Applied to a single column:

            import liken as lk

            pipeline = lk.pipeline().step(
                [
                    lk.col("email").exact(),
                    lk.col("name").sorted_tokens(),
                ]
            )

            df = (
                lk.dedupe(df)
                .apply(pipeline)
                .drop_duplicates(keep="last")
            )

            >>> df # before
            +------+--------------+---------------------+
            | id   |     name     |        email        |
            +------+--------------+---------------------+
            |  1   |  John Smith  |  fizzpop@yahoo.com  |
            |  2   |  Smith John  |  fizzpop@yahoo.com  |
            |  3   |  Jane Doe    |   foobar@gmail.com  |
            |  4   |     null     |   foobar@gmail.com  |
            |  5   |     null     |   foobar@gmail.com  |
            +------+--------------+---------------------+

            >>> df # after
            +------+--------------+---------------------+
            | id   |     name     |        email        |
            +------+--------------+---------------------+
            |  2   |  Smith John  |  fizzpop@yahoo.com  | # rows 1, 2 same tokens
            |  3   |  Jane Doe    |   foobar@gmail.com  |
            |  5   |     null     |   foobar@gmail.com  | # nulls group together
            +------+--------------+---------------------+
    """
    return SortedTokens()
