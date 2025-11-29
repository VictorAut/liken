"""Perform near deduplication with fuzzywuzzy string matching"""

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


# STR STARTS WITH:


class StrStartsWith(DeduplicationStrategy):
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
        """Deduplicate records starting with the given pattern"""
        logger.debug(f'Deduping attribute "{attr}" with {self.__class__.__name__}' f"(pattern={self._pattern})")

        def _matches(value: str) -> bool:
            return (
                value.startswith(self._pattern)
                #
                if self._case
                else value.lower().startswith(self._pattern.lower())
            )

        unique_attr: np.ndarray = np.unique(self.wrapped_df.get_col(attr))
        match_map: dict = get_matches(_matches, unique_attr)
        new_attr: SeriesLike = self.wrapped_df.map_dict(attr, match_map)
        self.wrapped_df.put_col(TMP_ATTR_LABEL, new_attr)

        return self.assign_group_id(TMP_ATTR_LABEL, include_exact=False).drop_col(TMP_ATTR_LABEL)


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
        """Deduplicate records ending with the given pattern"""
        logger.debug(f'Deduping attribute "{attr}" with {self.__class__.__name__}' f"(pattern={self._pattern})")

        def _matches(value: str) -> bool:
            return (
                value.endswith(self._pattern)
                #
                if self._case
                else value.lower().endswith(self._pattern.lower())
            )

        unique_attr: np.ndarray = np.unique(self.wrapped_df.get_col(attr))
        match_map: dict = get_matches(_matches, unique_attr)
        new_attr: SeriesLike = self.wrapped_df.map_dict(attr, match_map)
        self.wrapped_df.put_col(TMP_ATTR_LABEL, new_attr)

        return self.assign_group_id(TMP_ATTR_LABEL, include_exact=False).drop_col(TMP_ATTR_LABEL)


# STR CONTAINS:


class StrContains(DeduplicationStrategy):
    """Strings contains deduper.

    Defaults to case sensitive. Supports literal substring or regex search.
    """

    def __init__(self, pattern: str, case: bool = True, regex: bool = False):
        super().__init__(pattern=pattern, case=case, regex=regex)
        self._pattern = pattern
        self._case = case
        self._regex = regex

        if self._regex:
            flags = 0 if self._case else re.IGNORECASE
            self._compiled_pattern = re.compile(self._pattern, flags)
        else:
            self._compiled_pattern = None

    @override
    def dedupe(self, attr: str, /) -> WrappedDataFrame:
        """Deduplicate records containing the given pattern."""
        logger.debug(
            f'Deduping attribute "{attr}" with {self.__class__.__name__}'
            f"(pattern={self._pattern}, regex={self._regex}, case={self._case})"
        )

        def _matches(value: str) -> bool:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return False

            if self._regex:
                return bool(self._compiled_pattern.search(value))
            else:
                if self._case:
                    return self._pattern in value
                else:
                    return self._pattern.lower() in value.lower()

        unique_attr: np.ndarray = np.unique(self.wrapped_df.get_col(attr))
        match_map: dict = get_matches(_matches, unique_attr)
        new_attr: SeriesLike = self.wrapped_df.map_dict(attr, match_map)
        self.wrapped_df.put_col(TMP_ATTR_LABEL, new_attr)

        return self.assign_group_id(TMP_ATTR_LABEL, include_exact=False).drop_col(TMP_ATTR_LABEL)


# HELPERS:


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
