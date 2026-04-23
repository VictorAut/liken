"""Define custom dedupers"""

from functools import wraps
from typing import Callable

from liken._custom import Custom
from liken._custom import PairGenerator
from liken.core.registries import dedupers_registry


def register(f: PairGenerator) -> Callable:
    """Register a custom function as a deduper.

    Custom functions can be registered for use as dedupers recognised by the
    ``Dedupe`` class. Use ``register`` as a decorator around the custom callable.

    The custom callable must accept a generic array-like object representing the
    contents of one or more DataFrame columns. The concrete column backing this
    array is resolved only when the deduper is applied.

    The expected function signature is:

        function(array, **kwargs)

    Args:
        f: A custom callable that returns integer pairs of indices identifying
            similar pairs in an array. Accepted callables are functions or
            generators, where generators are preferred.

    Returns:
        Callable

    Raises:
        TypeError: If any positional arguments are used when calling the
            registered deduper.

    Example:
        Registering a custom deduper

            import liken as lk

            @lk.custom.register
            def custom_deduper(array, **kwargs):
                # your code here
                yield ...

            df = (
                lk.dedupe(df)
                .apply(custom_deduper(**kwargs))
                .drop_duplicates("address")
            )

        E.g. the following Custom exact string-length deduplication deduper:

            @lk.custom.register
            def eq_str_len(array):
                n = len(array)
                for i in range(n):
                    for j in range(i + 1, n):
                        if len(array[i]) == len(array[j]):
                            yield i, j

        Applying the deduper:

            df = (
                lk.dedupe(df)
                .apply(eq_str_len()) # array arg implicitely passed
                .drop_duplicates("address")
            )

        Before:

            +------+-----------+
            | id   | address   |
            +------+-----------+
            |  1   | london    |
            |  2   | paris     |
            |  3   | tokyo     |
            +------+-----------+

        "tokyo" and "paris" have the same length, so reduced:

            +------+-----------+
            | id   | address   |
            +------+-----------+
            |  1   | london    |
            |  2   | paris     |
            +------+-----------+

        Keyword-only enforcement:

            Deduper(df).apply(my_func(is_upper_caps=True))  # OK
            Deduper(df).apply(my_func(True))                # Raises TypeError
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        if args:
            raise TypeError(f"{f.__name__} must be called with keyword arguments only")
        return Custom(f, **kwargs)

    # Add to registry
    dedupers_registry.register(f"{f.__name__}", func=wrapper)

    return wrapper
