"""Perform near deduplication with fuzzywuzzy string matching"""

import logging
from typing_extensions import override

import numpy as np

from dupegrouper.definitions import TMP_ATTR_LABEL, SeriesLike
from dupegrouper.wrappers import WrappedDataFrame
from dupegrouper.strategy import DeduplicationStrategy


# LOGGER:


logger = logging.getLogger(__name__)


# STR ENDS WITH:


class StrEndsWith(DeduplicationStrategy):
    """Strings start with deduper.

    Defaults to case sensitive.

    Regex is not supported, please use `StrContains` otherwise.
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
        """Deduplicate records ending with the given pattern
        """
        logger.debug(f'Deduping attribute "{attr}" with {self.__class__.__name__}' f"(pattern={self._pattern})")

        def _endswith_case(value: str) -> bool:
            return (
                value.endswith(self._pattern)
                #
                if self._case
                else value.lower().endswith(self._pattern.lower())
            )

        match_map = {}
        for key in (col := np.unique(self.wrapped_df.get_col(attr))):
            for value in col:
                key_starts = _endswith_case(key)
                value_starts = _endswith_case(value)

                if key_starts and value_starts:
                    match_map[key] = value
                    break

        new_attr: SeriesLike = self.wrapped_df.map_dict(attr, match_map)

        self.wrapped_df.put_col(TMP_ATTR_LABEL, new_attr)

        return self.assign_group_id(TMP_ATTR_LABEL, include_exact=False).drop_col(TMP_ATTR_LABEL)