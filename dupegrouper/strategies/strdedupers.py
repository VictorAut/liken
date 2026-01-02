import logging
import re
import typing
from typing_extensions import override

import numpy as np

from dupegrouper.definitions import TMP_ATTR_LABEL, SeriesLike
from dupegrouper.wrappers import WrappedDataFrame
from dupegrouper.strategy import DeduplicationStrategy


# LOGGER:


logger = logging.getLogger(__name__)


# GENERIC STRING METHODS:


class StringMethods(DeduplicationStrategy):
    """TODO"""

    def __init__(self, pattern: str, case: bool = True):
        self._pattern = pattern
        self._case = case

    def _matches(self, value):
        del value  # Unused
        pass

    @staticmethod
    def get_matches(
        match_fn: typing.Callable[[str], bool],
        attr: np.ndarray,
    ) -> dict[str, str]:
        match_map = {}
        for key in attr:
            for value in attr:
                if match_fn(key) and match_fn(value):
                    match_map[key] = value
                    break
        return match_map
    
    @override
    def dedupe(self, attr: str, /) -> WrappedDataFrame:

        unique_attr: np.ndarray = np.unique(self.wrapped_df.get_col(attr))
        match_map: dict = self.get_matches(self._matches, unique_attr)
        new_attr: SeriesLike = self.wrapped_df.map_dict(attr, match_map)
        self.wrapped_df.put_col(TMP_ATTR_LABEL, new_attr)

        return self.assign_canonical_id(TMP_ATTR_LABEL, include_exact=False).drop_col(TMP_ATTR_LABEL)


# STR STARTS WITH:


class StrStartsWith(StringMethods):
    """Strings start with deduper.

    Defaults to case sensitive.

    Regex is not supported, please use `StrContains` otherwise.
    """

    @override
    def _matches(self, value: str) -> bool:
        return (
            value.startswith(self._pattern)
            #
            if self._case
            else value.lower().startswith(self._pattern.lower())
        )


# STR ENDS WITH:


class StrEndsWith(StringMethods):
    """Strings start with deduper.

    Defaults to case sensitive.

    Regex is not supported, please use `StrContains` otherwise.
    """

    @override
    def _matches(self, value: str) -> bool:
        return (
            value.endswith(self._pattern)
            #
            if self._case
            else value.lower().endswith(self._pattern.lower())
        )


# STR CONTAINS:


class StrContains(StringMethods):
    """Strings contains deduper.

    Defaults to case sensitive. Supports literal substring or regex search.
    """

    def __init__(self, pattern: str, case: bool = True, regex: bool = False):
        super().__init__(pattern=pattern, case=case)
        self._regex = regex

        if self._regex:
            flags = 0 if self._case else re.IGNORECASE
            self._compiled_pattern = re.compile(self._pattern, flags)

    @override
    def _matches(self, value: str) -> bool:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return False

        if self._regex:
            return bool(self._compiled_pattern.search(value))
        else:
            if self._case:
                return self._pattern in value
            else:
                return self._pattern.lower() in value.lower()
