from functools import wraps
from typing import Callable

from .._custom import Custom


# REGISTER:


def register(f: Callable):
    """TODO"""
    @wraps(f)
    def wrapper(**kwargs):
        return Custom(f, **kwargs)

    return wrapper
