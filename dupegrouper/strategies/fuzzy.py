"""Perform near deduplication with fuzzywuzzy string matching"""

import functools
import logging
from typing_extensions import override

import numpy as np
from rapidfuzz import fuzz

from dupegrouper.definitions import TMP_ATTR, SeriesLike
from dupegrouper.wrappers import WrappedDataFrame
from dupegrouper.strategy import DeduplicationStrategy


# LOGGER:


logger = logging.getLogger(__name__)


# FUZZY:


class Fuzzy(DeduplicationStrategy):

    def __init__(self, tolerance: float = 0.05):
        super().__init__(tolerance=tolerance)
        self._tolerance = tolerance
        self._ratio = 100 * (1 - tolerance)

    @staticmethod
    @functools.cache
    def _fuzz_ratio(s1, s2) -> float:
        return fuzz.ratio(s1, s2)

    @override
    def dedupe(self, attr: str, /) -> WrappedDataFrame:
        """Deduplicate with string match using fuzzy wuzzy

        String matches are applied on only *unique* instances of the attribute,
        for optimization. fuzzy wuzzy matches are cached, optimising
        computation of matches for instances of frequent duplication.
        """
        logger.debug(f'Deduping attribute "{attr}" with {self.__class__.__name__}' f"(tolerance={self._tolerance})")

        uattrs = np.unique(self.wrapped_df.get_col(attr))

        similarity_matrix = np.array([[self._fuzz_ratio(s1, s2) for s1 in uattrs] for s2 in uattrs])

        match_indices = np.where(similarity_matrix > self._ratio)

        fuzzy_map: dict[str, str] = {uattrs[i]: uattrs[j] for i, j in zip(*match_indices)}

        attr_map: SeriesLike = self.wrapped_df.map_dict(attr, fuzzy_map)

        self.wrapped_df.put_col(TMP_ATTR, attr_map)

        return self.assign_group_id(TMP_ATTR).drop_col(TMP_ATTR)
