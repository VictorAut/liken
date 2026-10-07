import warnings
from typing import final


# EXCEPTIONS:


@final
class InvalidDeduperError(TypeError):
    def __init__(self, msg):
        super().__init__(msg)


def warn(msg: str) -> None:
    """Emit a `UserWarning` with the given message."""
    return warnings.warn(msg, category=UserWarning, stacklevel=2)
