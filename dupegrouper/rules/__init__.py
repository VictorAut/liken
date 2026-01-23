from .._strats_library import str_contains, str_endswith, str_startswith
from .._strats_manager import Rules, on

_rules_api = ["Rules", "on"]
_rules_strats = [
    "str_contains",
    "str_endswith",
    "str_startswith",
]

__all__ = _rules_api + _rules_strats