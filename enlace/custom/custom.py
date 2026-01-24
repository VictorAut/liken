from typing import Callable

from .._custom import register as _register


def register(f: Callable):
    """TODO"""
    return _register(f)
