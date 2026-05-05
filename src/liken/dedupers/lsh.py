"""lsh deduper"""

from collections.abc import Iterator
from typing import ClassVar
from typing import final

import pyarrow as pa
from datasketch import MinHash
from datasketch import MinHashLSH

from liken.core.deduper import BaseDeduper
from liken.core.deduper import SingleColumnMixin
from liken.core.deduper import ThresholdDeduper
from liken.core.registries import dedupers_registry
from liken.types import SimilarPairIndices


@final
class LSH(
    SingleColumnMixin,
    ThresholdDeduper,
):
    """
    Locality Sensitive Hashing deduper
    """

    _NAME: ClassVar[str] = "lsh"

    def __init__(
        self,
        ngram: int = 3,
        num_perm: int = 128,
        threshold: float = 0.95,
    ):
        super().__init__(
            threshold=threshold,
            ngram=ngram,
            num_perm=num_perm,
        )
        self._ngram = ngram
        self._threshold = threshold
        self._num_perm = num_perm

    def _gen_token(self, text: pa.Scalar) -> Iterator:
        for i in range(len(text) - self._ngram + 1):
            yield text[i : i + self._ngram]

    def _build_minhashes(self, array: list) -> list[MinHash]:
        minhashes: list[MinHash] = []
        for value in array:
            m = MinHash(num_perm=self._num_perm)
            for token in self._gen_token(value):
                m.update(token.encode("utf8"))
            minhashes.append(m)
        return minhashes

    def _lsh(self, minhashes: list[MinHash]) -> MinHashLSH:
        lsh = MinHashLSH(
            threshold=self._threshold,
            num_perm=self._num_perm,
        )

        for i, m in enumerate(minhashes):
            lsh.insert(i, m)

        return lsh

    def _gen_similarity_pairs(self, array: pa.Array) -> Iterator[SimilarPairIndices]:
        array: list = array.to_pylist()

        minhashes: list[MinHash] = self._build_minhashes(array)
        lsh: MinHashLSH = self._lsh(minhashes)

        for idx, minhash in enumerate(minhashes):
            for idy in lsh.query(minhash):
                if idx < idy:
                    yield idx, idy

    def __str__(self):
        return self.str_representation(self._NAME)


@dedupers_registry.register("lsh")
def lsh(
    threshold: float = 0.95,
    ngram: int = 3,
    num_perm: int = 128,
) -> BaseDeduper:
    """Near string deduplication using locality sensitive hashing (LSH).

    Usage is on single columns of a dataframe. `lsh` is a tuneable deduper.
    Experimentation is required for optimal use.

    Args:
        threshold: the minimum threshold at which similarity between two pairs
            of values will be considered valid for deduplication.
        ngram: the number of character ngrams to consider. For `lsh`, and
            unlike the `tfidf` implementation, this is single integer ngram
            number. So, `ngram=1` is only unigrams. Increasing ngrams reduces
            overall deduplication. However, too small an `ngram` may result in false
            positives.
        num_perm: the number of MinHash permutations used to approximate
            similarity. Increasing this generally produces better matches, at
            greater computational cost. Very low numbers of permutations (< 64)
            can produce unreliable results.

    Returns:
        Instance of `BaseDeduper`.

    Example:
        Applied to a single column:

            import liken as lk

            df = (
                lk.dedupe(df)
                .apply({"address": lsh(threshold=0.8, ngram=1)})
                .drop_duplicates(keep="last")
            )

        E.g.

            >>> df
            +------+-----------+----------------------+
            | id   |  address  |         email        |
            +------+-----------+----------------------+
            |  1   |  london   |  fizzpop@gmail.com   |
            |  2   |   null    |  foobar@gmail.com    |
            |  3   |  london   |  foobar@gmail.co.uk  |
            +------+-----------+----------------------+

            >>> df # After deduplication
            +------+-----------+----------------------+
            | id   |  address  |         email        |
            +------+-----------+----------------------+
            |  1   |  london   |  fizzpop@gmail.com   |
            |  3   |  london   |  foobar@gmail.co.uk  |
            +------+-----------+----------------------+
    """
    return LSH(threshold=threshold, ngram=ngram, num_perm=num_perm)
