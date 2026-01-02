"""Near deduplication with cosine distance"""

import hashlib
import logging
from typing_extensions import override
import typing

import numpy as np
from numpy.linalg import norm

from dupegrouper.definitions import HASH_ATTR_LABEL, TMP_ATTR_LABEL, SeriesLike
from dupegrouper.wrappers import WrappedDataFrame
from dupegrouper.strategy import DeduplicationStrategy


# LOGGER:


logger = logging.getLogger(__name__)


# COMPOUND COLUMN DEDUPER


class CompoundColumn(DeduplicationStrategy):

    def __init__(self, tolerance: float = 0.05):
        super().__init__(tolerance=tolerance)
        self._threshold = 1 - tolerance

    @staticmethod
    def _hash(value: typing.Any) -> str:
        """deterministic hash for reproducability; order sensitive"""
        return hashlib.sha256(value.tobytes()).hexdigest()

    def _gen_similarities(self, attrs):
        del attrs # Unused
        pass

    @override
    def dedupe(self, attrs: typing.Iterable[str], /) -> WrappedDataFrame:
        """TODO"""

        attrs = np.asarray(self.wrapped_df.get_cols(attrs))

        attrs_hash: SeriesLike = [self._hash(i) for i in attrs]

        self.wrapped_df.put_col(HASH_ATTR_LABEL, attrs_hash)

        for idx, idy in self._gen_similarities(attrs):
            # skip if equal
            if attrs_hash[idx] == attrs_hash[idy]:
                continue

            indice_map: dict[str, str] = {attrs_hash[idx]: attrs_hash[idy]}
            attr_map: SeriesLike = self.wrapped_df.map_dict(HASH_ATTR_LABEL, indice_map)
            new_attr: SeriesLike = self.wrapped_df.fill_na(attr_map, self.wrapped_df.get_col(HASH_ATTR_LABEL))
            self.wrapped_df.put_col(TMP_ATTR_LABEL, new_attr)
            self.assign_canonical_id(TMP_ATTR_LABEL)
            self.wrapped_df.drop_col(TMP_ATTR_LABEL)
        
        return self.wrapped_df.drop_col(HASH_ATTR_LABEL)


# JACCARD:


class Jaccard(CompoundColumn):

    def _gen_similarities(self, attrs: np.ndarray) -> typing.Iterator[tuple[int, int]]:
        sets = [set(row) for row in attrs]
        
        n = len(attrs)
        for idx in range(n):
            for idy in range(idx+1, n):
                intersection = sets[idx] & sets[idy]

                if not intersection:
                    continue # no match
                
                union = sets[idx] | sets[idy]

                if not union:
                    continue # zero div: guardrail
                
                if len(intersection) / len(union) > self._threshold:
                    yield idx, idy


# COSINE:


class Cosine(CompoundColumn):

    def _gen_similarities(self, attrs: np.ndarray) -> typing.Iterator[tuple[int, int]]:  
        n = len(attrs)
        for idx in range(n):
            for idy in range(idx+1, n):
                product = np.dot(attrs[idx], attrs[idy])

                if not product:
                    continue # no match

                norms = norm(attrs[idx]) * norm(attrs[idy])

                if not norms:
                    continue # zero div: guardrail
                
                if product / norms > self._threshold:
                    yield idx, idy