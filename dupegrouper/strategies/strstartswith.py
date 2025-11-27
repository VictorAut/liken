"""Perform near deduplication with fuzzywuzzy string matching"""

import logging
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
            match_map[i] = i
            for j in col:
                key_starts = i.startswith(self._pattern) if self._case else i.lower().startswith(self._pattern.lower())
                value_starts = i.startswith(self._pattern) if self._case else j.lower().startswith(self._pattern.lower())

                if key_starts and value_starts:
                    match_map[i] = j
                    break

        new_attr: SeriesLike = self.wrapped_df.map_dict(attr, match_map)

        self.wrapped_df.put_col(TMP_ATTR_LABEL, new_attr)

        return self.assign_group_id(TMP_ATTR_LABEL).drop_col(TMP_ATTR_LABEL)

# _pattern = "calle"
# _case = False
# match_map = {}
# for i in (col := np.unique(df['address'])):
#     # match_map[i] = i
#     for j in col:
#         print(i, j)
#         key_starts = i.startswith(_pattern) if _case else i.lower().startswith(_pattern.lower())
#         value_starts = i.startswith(_pattern) if _case else j.lower().startswith(_pattern.lower())

#         if key_starts and value_starts:
#             match_map[i] = j
#             break

# df['address2'] = df['address'].map(match_map)

# # ####

# attrs = np.asarray(list(df['address2']))
# attrs = np.array([np.nan if x is None else x for x in attrs])  # handle full None lists
# groups = np.asarray(df['group_id'])

# unique_attrs, unique_indices = np.unique(
#     attrs,
#     return_index=True
# )

# first_groups = groups[unique_indices]

# attr_group_map = dict(zip(unique_attrs, first_groups))

# # iteratively: attrs -> value param; groups -> default param
# new_groups: np.ndarray = np.vectorize(
#     lambda value, default: attr_group_map.get(value, default),
# )(attrs, groups)