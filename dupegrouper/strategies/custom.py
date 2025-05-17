"""Perform near deduplication with a custom defined callable

Applies if the end user chooses to create their own function for deduplication
"""

import logging
import typing
from typing_extensions import override

from dupegrouper.definitions import TMP_ATTR
from dupegrouper.wrappers import WrappedDataFrame
from dupegrouper.strategy import DeduplicationStrategy


# LOGGER:


logger = logging.getLogger(__name__)


# TYPES:


_T = typing.TypeVar("_T")


# CUSTOM:


class Custom(DeduplicationStrategy):

    def __init__(
        self,
        func: typing.Callable[..., dict[_T, _T]],
        attr: str,
        /,
        **kwargs,
    ):
        super().__init__(
            func,
            attr,
            **kwargs,
        )
        self._func = func
        self._attr = attr
        self._kwargs = kwargs

    @override
    def dedupe(self, attr=None) -> WrappedDataFrame:
        """dedupe with custom defined callable

        Implements deduplication using a function defined _outside_ of the
        scope of this library i.e. by the end user.

        The function signature must be of the following style:

        `my_func(df, attr, /, **kwargs)`

        Where `df` is the dataframe, `attr` is a string identifying the label
        of the dataframe attribute requiring deduplication and kwargs are any
        number of additional keyword arguments taken by the function

        `df` and `attr`, must be *positional* arguments in the correct order!
        """
        del attr  # Unused: initialised as class private attribute
        logger.debug(
            f'Deduping attribute "{self._attr}" with {self._func.__name__}'
            f'({", ".join(f"{k}={v}" for k, v in self._kwargs.items())})'
        )

        attr_map = self.wrapped_df.map_dict(
            self._attr,
            self._func(
                self.wrapped_df,
                self._attr,
                **self._kwargs,
            ),
        )

        self.wrapped_df.put_col(TMP_ATTR, attr_map)

        return self.assign_group_id(TMP_ATTR).drop_col(TMP_ATTR)
