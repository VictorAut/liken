import re
import logging
from typing_extensions import override
import numpy as np

from dupegrouper.definitions import TMP_ATTR_LABEL, SeriesLike
from dupegrouper.wrappers import WrappedDataFrame
from dupegrouper.strategy import DeduplicationStrategy

logger = logging.getLogger(__name__)


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

        match_map = {}
        # Get unique values
        for key in np.unique(self.wrapped_df.get_col(attr)):
            for value in np.unique(self.wrapped_df.get_col(attr)):
                if _matches(key) and _matches(value):
                    match_map[key] = value
                    break

        new_attr: SeriesLike = self.wrapped_df.map_dict(attr, match_map)
        self.wrapped_df.put_col(TMP_ATTR_LABEL, new_attr)

        return self.assign_group_id(TMP_ATTR_LABEL, include_exact=False).drop_col(TMP_ATTR_LABEL)
