"""Perform near deduplication with fuzzywuzzy string matching"""

import logging
from typing_extensions import override

import numpy as np

from dupegrouper.definitions import TMP_ATTR, SeriesLike
from dupegrouper.wrappers import WrappedDataFrame
from dupegrouper.strategy import DeduplicationStrategy


# LOGGER:


logger = logging.getLogger(__name__)


# STR STARTS WITH:


class StrStartsWith(DeduplicationStrategy):
    """Strings start with deduper.

    Defaults to case sensitive.
    """

    def __init__(self, pattern: str, case: bool = True):
        super().__init__(
            pattern=pattern,
            case=case,
        )
        self._pattern = pattern
        self._case = case

    @override
    def dedupe(self, attr: str, /) -> WrappedDataFrame:
        """Deduplicate records starting with the given pattern

        String matches are applied on only *unique* instances of the attribute,
        for optimization. fuzzy wuzzy matches are cached, optimising
        computation of matches for instances of frequent duplication.
        """
        logger.debug(f'Deduping attribute "{attr}" with {self.__class__.__name__}' f"(pattern={self._pattern})")

        match_map = {}
        for i in (col := np.unique(self.wrapped_df.get_col(attr))):
            # match_map[i] = i
            for j in col:
                key_starts = i.startswith(self._pattern) if self._case else i.lower().startswith(self._pattern)
                value_starts = i.startswith(self._pattern) if self._case else j.lower().startswith(self._pattern)

                if key_starts and value_starts:
                    match_map[i] = j
                    break

        attr_map: SeriesLike = self.wrapped_df.map_dict(attr, match_map)

        self.wrapped_df.put_col(TMP_ATTR, attr_map)

        return self.assign_group_id(TMP_ATTR).drop_col(TMP_ATTR)
