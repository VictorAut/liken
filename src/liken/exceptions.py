import warnings
from typing import final


# EXCEPTIONS:


class InvalidArgumentError(Exception):
    """Base class for every liken argument-validation error."""


@final
class InvalidArgumentTypeError(InvalidArgumentError, TypeError):
    """An argument has the wrong type."""


@final
class InvalidArgumentValueError(InvalidArgumentError, ValueError):
    """An argument has an unacceptable value despite a correct type."""


class InvalidDeduperError(InvalidArgumentError, TypeError):
    """A deduper, or collection of dedupers, is misconfigured."""

    def __init__(self, msg):
        super().__init__(msg)


def warn(msg: str, *, stacklevel: int = 3) -> None:
    """Emit a `UserWarning` with the given message.

    The default stack level attributes the warning to the user's call when
    `warn` sits one liken frame below it. Callers further down the stack
    pass a higher level.
    """
    return warnings.warn(msg, category=UserWarning, stacklevel=stacklevel)
