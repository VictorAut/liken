"""Defines collections of dedupers"""

from __future__ import annotations

from copy import deepcopy
from typing import final

from liken.collections.dict import DeduplicationDict
from liken.collections.pipelines import Col
from liken.collections.pipelines import Pipeline
from liken.constants import INVALID_FALLBACK_MSG
from liken.constants import INVALID_SEQUENCE_AFTER_DICT_MSG
from liken.constants import SEQUENTIAL_API_DEFAULT_KEY
from liken.constants import WARN_DICT_REPLACES_SEQUENCE_MSG
from liken.constants import WARN_RULES_REPLACES_RULES_MSG
from liken.core.deduper import BaseDeduper
from liken.exceptions import InvalidDeduperError
from liken.exceptions import warn


# COLLECTIONS MANAGER:


@final
class CollectionsManager:
    """
    Manage and validate collection(s) of dedupers.

    Supports addition of dedupers as part of the three APIs:
    - Sequential
    - Dict
    - Pipeline

    For Sequential dedupers, as instances of `BaseDeduper` are sequentially
    to an idential structure of the Dict API but under a single default
    dictionary key. Keys are columns names, and values are iterables of
    dedupers.

    Raises:
        InvalidDeduperError for any misconfigured addition of a deduper.
    """

    def __init__(self) -> None:
        self._dedupers: DeduplicationDict | Pipeline = DeduplicationDict({SEQUENTIAL_API_DEFAULT_KEY: []})
        self.has_applies: bool = False

    @property
    def is_sequential_applied(self) -> bool:
        """checks to see if any dedupers exist in the default key"""
        if isinstance(self._dedupers, Pipeline):
            return False
        return set(self._dedupers) == {SEQUENTIAL_API_DEFAULT_KEY}

    def apply(self, deduper: BaseDeduper | dict | DeduplicationDict | Pipeline) -> None:
        """Loads a deduper / collection of dedupers into the manager

        This function currently handles all possible instances of dedupers, and
        the implementation achieves this by writing to the deduplication
        dictionary or overwriting the dictionary with `Pipeline`.

        If the input deduper is `BaseDeduper` then "Sequential" API is in use. If
        dict (or DeduplicationDict — even though this is not public) then it is the
        "Dict" API. Else "Pipeline" API is in use.

        Note also that as Pipeline contains Col and combinations of Col operated
        with & results in self mutation, need deep copy to allow for
        serialization to Spark workers."""

        # track that at least one apply made
        # if not, used by `Dedupe` to include an exact deduper by default
        self.has_applies = True

        if isinstance(deduper, BaseDeduper):
            if not self.is_sequential_applied:
                raise InvalidDeduperError(INVALID_SEQUENCE_AFTER_DICT_MSG)
            self._dedupers[SEQUENTIAL_API_DEFAULT_KEY].append(deduper)  # type: ignore
            return

        if isinstance(deduper, dict | DeduplicationDict):
            if self._dedupers[SEQUENTIAL_API_DEFAULT_KEY]:  # type: ignore
                warn(WARN_DICT_REPLACES_SEQUENCE_MSG)
            self._dedupers = DeduplicationDict(deduper)
            return

        if isinstance(deduper, Col):
            deduper = Pipeline().step(deduper)

        if isinstance(deduper, Pipeline):
            if isinstance(self._dedupers, Pipeline):
                warn(WARN_RULES_REPLACES_RULES_MSG)
            # required for spark serialization
            self._dedupers = deepcopy(deduper)  # type: ignore
            return

        raise InvalidDeduperError(INVALID_FALLBACK_MSG.format(type(deduper).__name__))

    def get(self) -> DeduplicationDict | Pipeline:
        return self._dedupers

    def pretty_get(self) -> None | str:
        """string representation of dedupers.

        Output string must be formatted approximately such that it can be used
        with .apply(), i.e. a string representation of one of:
            - BaseDeduper
            - DeduplicationDict
            - Pipeline
        The seuqneital API with numerous additions of BaseStraegy means there
        is not good way to retried this such that is available to "apply". So,
        default to returning it as a list representation.
        """
        dedupers = self.get()

        if isinstance(dedupers, DeduplicationDict):
            if self.is_sequential_applied:
                deduper: list = dedupers[SEQUENTIAL_API_DEFAULT_KEY]
                if not deduper:
                    return None
                return str(*deduper)
            return str(dedupers)
        return str(dedupers)

    def reset(self):
        """Reset collection to empty defaultdict"""
        self._dedupers = DeduplicationDict({SEQUENTIAL_API_DEFAULT_KEY: []})
