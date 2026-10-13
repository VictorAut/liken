from __future__ import annotations

import os
from typing import Final


# Default canonical_id label in the dataframe
CANONICAL_ID: Final[str] = os.environ.get("CANONICAL_ID", "canonical_id")

# Sequential API use will load to this dictionary key by default:
SEQUENTIAL_API_DEFAULT_KEY: Final[str] = "_default_"


# ERROR MESSAGES

# collection errors

INVALID_DICT_KEY_MSG: Final[str] = "Invalid type for dict key type: expected str or tuple, got '{}'"
INVALID_DICT_VALUE_MSG: Final[str] = "Invalid type for dict value: expected list, tuple or 'BaseDeduper', got '{}'"
INVALID_DICT_MEMBER_MSG: Final[str] = (
    "Invalid type for dict value member: at index {} for key '{}': 'expected 'BaseDeduper', got '{}'"
)
INVALID_SEQUENCE_AFTER_DICT_MSG: Final[str] = (
    "Cannot apply a 'BaseDeduper' after a deduper mapping (dict) has been set. "
    "Use either individual 'BaseDeduper' instances or a dict of dedupers, not both."
)
INVALID_FALLBACK_MSG: Final[str] = "Invalid deduper: Expected a 'BaseDeduper', a dict or 'Pipeline', got '{}'"

# collection warnings

WARN_DICT_REPLACES_SEQUENCE_MSG: Final[str] = "Replacing previously added sequence deduper with a dict deduper"
WARN_RULES_REPLACES_RULES_MSG: Final[str] = (
    "Replacing previously added 'Pipeline' deduper with a new 'Pipeline' deduper"
)
