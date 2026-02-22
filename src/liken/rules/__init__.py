"""The Rules API defines liken's most flexible and powerful approach.

With this API you can construct rules that combine strategies using and
statements.

Additional boolean choice strategies are defined here — they can be powerfully
combined with the `liken` standard deduplication strategies.
"""

from .._strats_library import isin
from .._strats_library import isna
from .._strats_library import str_contains
from .._strats_library import str_endswith
from .._strats_library import str_len
from .._strats_library import str_startswith
from .._strats_manager import Rules
from .._strats_manager import on


__all__ = [
    "Rules",
    "on",
    "isna",
    "isin",
    "str_startswith",
    "str_contains",
    "str_endswith",
    "str_len",
]
