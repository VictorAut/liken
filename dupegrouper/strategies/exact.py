"""Perform exact deduplication"""

import logging
from typing_extensions import override

from dupegrouper.wrappers import WrappedDataFrame
from dupegrouper.strategy import DeduplicationStrategy


# LOGGER:


logger = logging.getLogger(__name__)


# EXACT:


class Exact(DeduplicationStrategy):

    @override
    def dedupe(self, attr: str, /) -> WrappedDataFrame:
        logger.debug(f'Deduping attribute "{attr}" with {self.__class__.__name__}()')
        return self.assign_group_id(attr)
