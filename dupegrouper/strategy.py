"""Abstract base class for derived deduplication atrategies

This module contains `DeduplicationStrategy` which provides
`assign_group_id()`, which is at the core functionality of `dupegrouper` and is
used for any deduplication that requires *grouping*. Additionally, the
overrideable `dedupe()` is defined.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import logging
import typing

import numpy as np

from dupegrouper.definitions import GROUP_ID
from dupegrouper.wrappers import WrappedDataFrame


# LOGGER:


_logger = logging.getLogger(__name__)


# STRATEGY:


class DeduplicationStrategy(ABC):
    """Defines a deduplication strategy."""

    def __init__(self, *args, **kwargs):
        self._init_args = args
        self._init_kwargs = kwargs

    def reinstantiate(self):
        return self.__class__(*self._init_args, **self._init_kwargs)

    def with_frame(self, wrapped_df: WrappedDataFrame) -> typing.Self:
        """Inject dataframe data and load dataframe methods corresponding
        to the type of the dataframe the corresponding methods.

        Args:
            df: The dataframe to set

        Returns:
            self: i.e. allow for further chaining
        """
        self.wrapped_df: WrappedDataFrame = wrapped_df
        return self

    def assign_group_id(self, attr: str) -> WrappedDataFrame:
        """Assign new group ids according to duplicated instances of attribute.

        Array-like contents of the dataframe's attributes are collected as a
        numpy array, along with the group id. unique instances are found, and
        the *first* group id of that attribute is identified. This allows to
        then assign this "first" group id to all subsequent instances of a
        given unique attribute thus "flooring" the group ids.

        This implementation is akin to

            df.groupby(attr).transform("first").fill_null("group_id")

        Where the null backfill is implemented to handle instances where data
        in the attribute `attr` is incomplete — which happens in instances of
        iterative application of this function, or, when the function is
        applied to an attribute `attr` that contains only matches, i.e., a
        partial map of matches.

        Args:
            attr: the dataframe label of the attribute

        Returns:
            wrapped_df; i.e. an instance "WrappedDataFrame" i.e. container
            of data and linked dataframe methods; ready for further downstream
            processing.
        """
        _logger.debug(f'Re-assigning new "group_id" per duped instance of attribute "{attr}"')

        attrs = np.asarray(self.wrapped_df.get_col(attr))
        attrs = np.array([np.nan if x is None else x for x in attrs])  # handle full None lists
        groups = np.asarray(self.wrapped_df.get_col(GROUP_ID))

        unique_attrs, unique_indices = np.unique(
            attrs,
            return_index=True,
        )

        first_groups = groups[unique_indices]

        attr_group_map = dict(zip(unique_attrs, first_groups))

        # iteratively: attrs -> value param; groups -> default param
        new_groups: np.ndarray = np.vectorize(
            lambda value, default: attr_group_map.get(
                value,
                default,
            )
        )(
            attrs,
            groups,
        )

        return self.wrapped_df.put_col(GROUP_ID, new_groups)

    @abstractmethod
    def dedupe(self, attr: str) -> WrappedDataFrame:
        """Use `assign_group_id` to implement deduplication logic

        Args:
            attr: The attribute to use for deduplication.

        Returns:
            A deduplicated instance of WrappedDataFrame
        """
        pass  # pragma: no cover
