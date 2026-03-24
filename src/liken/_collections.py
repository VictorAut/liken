"""Defines collections of dedupers"""

from __future__ import annotations

from collections import UserDict
from copy import deepcopy
from typing import final

from liken._constants import INVALID_DICT_KEY_MSG
from liken._constants import INVALID_DICT_MEMBER_MSG
from liken._constants import INVALID_DICT_VALUE_MSG
from liken._constants import INVALID_FALLBACK_MSG
from liken._constants import INVALID_SEQUENCE_AFTER_DICT_MSG
from liken._constants import SEQUENTIAL_API_DEFAULT_KEY
from liken._constants import WARN_DICT_REPLACES_SEQUENCE_MSG
from liken._constants import WARN_RULES_REPLACES_RULES_MSG
from liken._dedupers import BaseDeduper
from liken._exceptions import InvalidDeduperError
from liken._exceptions import warn
from liken._pipelines import Col
from liken._pipelines import Pipeline


# DICT CONFIG:


@final
class DeduplicationDict(UserDict):
    """Dict collection for dedupers in the Sequential and Dict APIs

    For Sequential API all values (dedupers) are added under a default key.

    For Dict API column label(s) (i.e. str or tuple) are the keys."""

    def __setitem__(self, key, value):
        if not isinstance(key, str | tuple):
            raise InvalidDeduperError(INVALID_DICT_KEY_MSG.format(type(key).__name__))
        if not isinstance(value, list | tuple | BaseDeduper):
            raise InvalidDeduperError(INVALID_DICT_VALUE_MSG.format(type(value).__name__))
        if not isinstance(value, BaseDeduper):
            for i, member in enumerate(value):
                if not isinstance(member, BaseDeduper):
                    raise InvalidDeduperError(INVALID_DICT_MEMBER_MSG.format(i, key, type(member).__name__))
        else:
            value = (value,)
        super().__setitem__(key, value)

    def __str__(self):
        rep = ""
        for k, values in self.items():
            krep = ""
            for v in values:
                krep += "\n\t\t" + str(v) + ","
            rep += f"\n\t'{k}': ({krep[:-1]},\n\t\t),"
        return "{" + rep + "\n}"


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
            self._dedupers[SEQUENTIAL_API_DEFAULT_KEY].append(deduper)
            return

        if isinstance(deduper, dict | DeduplicationDict):
            if self._dedupers[SEQUENTIAL_API_DEFAULT_KEY]:
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
