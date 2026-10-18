"""Defines collections of dedupers"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING
from typing import final

from liken.collections.dict import DeduplicationDict
from liken.collections.pipelines import Col
from liken.collections.pipelines import Pipeline
from liken.constants import INVALID_APPLY_AFTER_PIPELINE_MSG
from liken.constants import INVALID_COLUMNS_NOT_NONE_MSG
from liken.constants import INVALID_COLUMNS_REQUIRED_MSG
from liken.constants import INVALID_EMPTY_DICT_MSG
from liken.constants import INVALID_EMPTY_DICT_VALUE_MSG
from liken.constants import INVALID_EMPTY_PIPELINE_MSG
from liken.constants import INVALID_FALLBACK_MSG
from liken.constants import INVALID_RESERVED_DICT_KEY_MSG
from liken.constants import INVALID_SEQUENCE_AFTER_DICT_MSG
from liken.constants import SEQUENTIAL_API_DEFAULT_KEY
from liken.constants import WARN_DICT_REPLACES_DICT_MSG
from liken.constants import WARN_DICT_REPLACES_PIPELINE_MSG
from liken.constants import WARN_DICT_REPLACES_SEQUENCE_MSG
from liken.constants import WARN_PIPELINE_REPLACES_DICT_MSG
from liken.constants import WARN_PIPELINE_REPLACES_SEQUENCE_MSG
from liken.constants import WARN_RULES_REPLACES_RULES_MSG
from liken.core.deduper import BaseDeduper
from liken.exceptions import InvalidArgumentValueError
from liken.exceptions import InvalidDeduperError
from liken.exceptions import warn


if TYPE_CHECKING:
    from liken.types import Columns


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
    to an identical structure of the Dict API but under a single default
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

        Note also that a deep copy of the Pipeline is stored to allow for
        serialization to Spark workers."""

        if isinstance(deduper, BaseDeduper):
            if isinstance(self._dedupers, Pipeline):
                raise InvalidDeduperError(INVALID_APPLY_AFTER_PIPELINE_MSG)
            if not self.is_sequential_applied:
                raise InvalidDeduperError(INVALID_SEQUENCE_AFTER_DICT_MSG)
            # The check above guarantees a `DeduplicationDict` here, but mypy
            # cannot narrow through `is_sequential_applied` (`index`); the
            # tuple member of the value union is only stored by the Dict API
            # path (`union-attr`).
            self._dedupers[SEQUENTIAL_API_DEFAULT_KEY].append(deduper)  # type: ignore[index, union-attr]
            # track that at least one apply succeeded; if not, used by
            # `Dedupe` to include an exact deduper by default. Set only after
            # validation, so a rejected apply cannot poison the flag into
            # skipping the exact fallback.
            self.has_applies = True
            return

        if isinstance(deduper, dict | DeduplicationDict):
            if not deduper:
                raise InvalidDeduperError(INVALID_EMPTY_DICT_MSG)
            for key, value in deduper.items():
                if key == SEQUENTIAL_API_DEFAULT_KEY:
                    raise InvalidDeduperError(INVALID_RESERVED_DICT_KEY_MSG.format(key))
                if isinstance(value, list | tuple) and not value:
                    raise InvalidDeduperError(INVALID_EMPTY_DICT_VALUE_MSG.format(key))
            # replace policy: every non-accumulating transition warns.
            if isinstance(self._dedupers, DeduplicationDict):
                if self.is_sequential_applied:
                    if self._dedupers[SEQUENTIAL_API_DEFAULT_KEY]:  # type: ignore[index, union-attr]
                        warn(WARN_DICT_REPLACES_SEQUENCE_MSG, stacklevel=4)
                else:
                    warn(WARN_DICT_REPLACES_DICT_MSG, stacklevel=4)
            else:
                warn(WARN_DICT_REPLACES_PIPELINE_MSG, stacklevel=4)
            self._dedupers = DeduplicationDict(deduper)
            self.has_applies = True
            return

        if isinstance(deduper, Col):
            deduper = Pipeline().step(deduper)

        if isinstance(deduper, Pipeline):
            if not deduper.steps:
                raise InvalidDeduperError(INVALID_EMPTY_PIPELINE_MSG)
            # replace policy: every non-accumulating transition warns.
            if isinstance(self._dedupers, Pipeline):
                warn(WARN_RULES_REPLACES_RULES_MSG, stacklevel=4)
            elif self.is_sequential_applied:
                if self._dedupers[SEQUENTIAL_API_DEFAULT_KEY]:  # type: ignore[index, union-attr]
                    warn(WARN_PIPELINE_REPLACES_SEQUENCE_MSG, stacklevel=4)
            else:
                warn(WARN_PIPELINE_REPLACES_DICT_MSG, stacklevel=4)
            # required for spark serialization
            self._dedupers = deepcopy(deduper)
            self.has_applies = True
            return

        raise InvalidDeduperError(INVALID_FALLBACK_MSG.format(type(deduper).__name__))

    def get(self) -> DeduplicationDict | Pipeline:
        return self._dedupers

    def validate_api_mode(self, columns: Columns | None) -> Columns | None:
        """Validate the `columns` argument against the api mode in use.

        Sequential mode — including the exact fallback — requires `columns`.
        Dict and pipeline modes require `columns` to be `None`, as the keys
        of the dict or the pipeline's columns define the deduplication.

        The type and structure of `columns` are validated first, by
        `liken.validations.validate_columns_arg`.

        Raises:
            InvalidArgumentValueError: for a combination not allowed by the
                api mode in use.
        """
        if self.is_sequential_applied:
            if columns is None or columns == ():
                raise InvalidArgumentValueError(INVALID_COLUMNS_REQUIRED_MSG)
        elif columns is not None:
            raise InvalidArgumentValueError(INVALID_COLUMNS_NOT_NONE_MSG)
        return columns

    def pretty_get(self) -> None | str:
        """string representation of dedupers.

        Output string is formatted such that it can be used with `.apply()`,
        i.e. a string representation of one of:
            - BaseDeduper
            - DeduplicationDict
            - Pipeline
        For the Sequential API with a single deduper, the deduper's own string
        representation is returned. With multiple sequential dedupers, the
        list representation is returned instead: there is no single-deduper
        string that is re-pasteable for `.apply()`.
        """
        dedupers = self.get()

        if isinstance(dedupers, DeduplicationDict):
            if self.is_sequential_applied:
                deduper: list[BaseDeduper] | tuple[BaseDeduper, ...] = dedupers[SEQUENTIAL_API_DEFAULT_KEY]
                if not deduper:
                    return None
                if len(deduper) == 1:
                    return str(deduper[0])
                return str(deduper)
            return str(dedupers)
        return str(dedupers)

    def reset(self):
        """Reset collection to empty defaultdict

        Also resets `has_applies`, so a subsequent enact applies the exact
        fallback again."""
        self._dedupers = DeduplicationDict({SEQUENTIAL_API_DEFAULT_KEY: []})
        self.has_applies = False
