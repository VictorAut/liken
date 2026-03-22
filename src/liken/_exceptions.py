import warnings
from typing import final


# EXCEPTIONS:


@final
class InvalidDeduperError(TypeError):
    def __init__(self, msg):
        super().__init__(msg)


# TODO: these warnings come up was "UserWarning". Change.
def warn(msg: str) -> None:
    return warnings.warn(msg, category=UserWarning)
