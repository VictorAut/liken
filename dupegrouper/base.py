"""dupegrouper main entrypoint

This module contains `Duped`, at the core of all 'dupe and group'
functionality provided by dupegrouper.
"""

from __future__ import annotations
from collections.abc import Iterator
from collections import defaultdict
from functools import singledispatchmethod
import inspect
import logging
from types import NoneType
from typing import cast, Literal

from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructField, StructType, DataType

from dupegrouper.constants import CANONICAL_ID, PYSPARK_TYPES
from dupegrouper.dataframe import (
    wrap,
    WrappedDataFrame,
    WrappedSparkDataFrame,
)
from dupegrouper.strats import BaseStrategy
from dupegrouper.types import DataFrameLike, StrategyMapCollection, Rule

from dupegrouper.custom import plugin_registry

print(plugin_registry)


# LOGGER:


_logger = logging.getLogger(__name__)


# BASE:


class Duped:
    """Top-level entrypoint for grouping duplicates

    This class handles initialisation of a dataframe, dispatching appropriately
    given the supported dataframe libraries (e.g. Pandas). An instance of this
    class can then accept a variety of strategies for deduplication and
    grouping.

    Upon initialisation, `Duped` sets a new column, usually `"canonical_id"`
    — but you can control this by setting an environment variable `CANONICAL_ID` at
    runtime. The canonical_id is a monotonically increasing, numeric id column
    starting at 1 to the length of the dataframe provided.
    """

    def __init__(
        self,
        df: DataFrameLike,
        spark_session: SparkSession | None = None,
        id: str | None = None,
        canonicalization_rule: Rule = "first",
    ):
        self._df: WrappedDataFrame = wrap(df, id)
        self._strategy_manager = StrategyManager()
        self._spark_session = spark_session
        self._id = id
        self._canonicalization_rule = canonicalization_rule

    def _call_strategy_canonicalizer(
        self,
        strategy: BaseStrategy,
        attr: str,
    ):
        """Dispatch the appropriate strategy deduplication method.

        If the strategy is an instance of a dupegrouper `BaseStrategy`
        the strategy will have been added as such, with it's parameters. In the
        case of a custom implementation of a Callable, passed as a tuple, we
        pass this *directly* to the `Custom` class and initialise that.

        Args:
            strategy: A `dupegrouper` deduplication strategy or a tuple
                containing a (customer) callable and its parameters.
            attr: The attribute used for deduplication.

        Returns:
            A deduplicated dataframe

        Raises:
            NotImplementedError.
        """
        return (
            strategy
            #
            .bind_frame(self._df)
            .bind_rule(self._canonicalization_rule)
            .canonicalize(attr)
        )

    @singledispatchmethod
    def _canonicalize(
        self,
        attr: str | None,
        strategies: StrategyMapCollection,
    ):
        """Dispatch the appropriate deduplication logic.

        If strategies have been added individually, they are stored under a
        "default" key and retrived as such when the public `.canonicalize` method is
        called _with_ the attribute label. In the case of having added
        strategies in one go with a direct dict (mapping) object, the attribute
        label is first extracted from strategy collection dictionary keys.
        Upon completing deduplication the strategy collection is wiped for
        (any) subsequent deduplication.

        Args:
            attr: The attribute used for deduplication; or None in the case
                of strategies being a mapping object

        Returns:
            None; internal `_df` attribute is updated.

        Raises:
            NotImplementedError.
        """
        del strategies  # Unused
        raise NotImplementedError(f"Unsupported attribute type: {type(attr)}")

    @_canonicalize.register(str | tuple)
    def _(self, attr, strategies):
        for strategy in strategies["default"]:
            self._df = self._call_strategy_canonicalizer(strategy, attr)

    @_canonicalize.register(NoneType)
    def _(self, attr, strategies):
        del attr  # Unused
        for attr, strategies in strategies.items():
            for strategy in strategies:
                self._df = self._call_strategy_canonicalizer(strategy, attr)

    def _canonicalize_spark(self, attr: str | None, strategies: StrategyMapCollection):
        """Spark specific deduplication helper

        Maps dataframe partitions to be processed via the RDD API yielding low-
        level list[Rows], which are then post-processed back to a dataframe.

        Args:
            attr: The attribute to deduplicate.
            strategies: the collection of strategies
        Retuns:
            Instance's _df attribute is updated
        """
        id = cast(str, self._id)
        rule = cast(str, self._canonicalization_rule)
        id_type = cast(DataType, PYSPARK_TYPES.get(dict(self._df.dtypes).get(id)))  # type: ignore

        canonicalized_rdd = self._df.rdd.mapPartitions(
            lambda partition_iter: _process_partition(
                partition_iter,
                strategies,
                id,
                attr,
                rule,
            )
        )

        if CANONICAL_ID in self._df.columns:
            schema = StructType(self._df.schema.fields)
        else:
            schema = StructType(self._df.schema.fields + [StructField(CANONICAL_ID, id_type, True)])

        self._df = WrappedSparkDataFrame(
            cast(SparkSession, self._spark_session).createDataFrame(canonicalized_rdd, schema=schema), id
        )

    # PUBLIC API:

    @singledispatchmethod
    def apply(self, strategy: BaseStrategy | StrategyMapCollection):
        """
        Add a strategy to the strategy manager.

        Instances of `BaseStrategy` are added to the
        "default" key. Mapping objects update the manager directly

        Args:
            strategy: A deduplication strategy or strategy collection
                (mapping) to add.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError(f"Unsupported strategy: {type(strategy)}")

    @apply.register(BaseStrategy)
    def _(self, strategy):
        self._strategy_manager.add("default", strategy)

    @apply.register(dict)
    def _(self, strategy: StrategyMapCollection):
        for attr, strat_list in strategy.items():
            for strat in strat_list:
                self._strategy_manager.add(attr, strat)

    def canonicalize(self, attr: str | None = None):
        """canonicalize, and group, the data based on the provided attribute

        Args:
            attr: The attribute to deduplicate. If strategies have been added
                as a mapping object, this must not passed, as the keys of the
                mapping object will be used instead
        """
        strategies = self._strategy_manager.get()

        if isinstance(self._df, WrappedSparkDataFrame):
            self._canonicalize_spark(attr, strategies)
        else:
            self._canonicalize(attr, strategies)

        self._strategy_manager.reset()

    @property
    def strategies(self) -> None | tuple[str, ...] | dict[str, tuple[str, ...]]:
        """
        Returns the strategies currently stored in the strategy manager.

        If no strategies are stored, returns `None`. Otherwise, returns a tuple
        of strategy names or a dictionary mapping attributes to their
        respective strategies.

        Returns:
            The stored strategies, formatted
        """
        strategies = self._strategy_manager.get()
        if not strategies:
            return None

        def parse_strategies(dict_values):
            return tuple(
                [
                    (vx[0].__name__ if isinstance(vx, tuple) else vx.__class__.__name__)
                    #
                    for vx in dict_values
                ]
            )

        if "default" in strategies:
            return tuple([parse_strategies(v) for _, v in strategies.items()])[0]
        return {k: parse_strategies(v) for k, v in strategies.items()}

    @property
    def df(self) -> DataFrameLike:
        return self._df.unwrap()


# STRATEGY MANAGER:


class StrategyManager:
    """
    Manage and validate collection(s) of deduplication strategies.

    Strategies are collected into a dictionary-like collection where keys are
    attribute names, and values are lists of strategies. Validation is provided
    upon addition allowing only the following stratgies types:
        - `BaseStrategy`
    A public property exposes stratgies upon successul addition and validation.
    A `StrategyTypeError` is thrown, otherwise.
    """

    def __init__(self) -> None:
        self._strategies: StrategyMapCollection = defaultdict(list)

    def add(
        self,
        attr_key: str,
        strategy: BaseStrategy | tuple,
    ):
        """Adds a strategy to the collection under a specific attribute key.

        Validates the strategy before adding it to the collection. If the
        strategy is not valid, a `StrategyTypeError` is raised.

        Args:
            attr_key: The key representing the attribute the strategy applies
                to.
            strategy: The deduplication strategy or a tuple containing a
                callable and its associated keyword arguments, as a mapping

        Raises:
            StrategyTypeError: If the strategy is not valid according to
            validation rules.
        """
        if self.validate(strategy):
            self._strategies[attr_key].append(strategy)  # type: ignore[attr-defined]
            return
        raise StrategyTypeError(strategy)

    def get(self) -> StrategyMapCollection:
        return self._strategies

    def validate(self, strategy) -> bool:
        """
        Validates a strategy

        The strategy to validate. Can be a `BaseStrategy`, a tuple, or
        a dict of the aforementioned strategies types i.e.
        dict[str, BaseStrategy | tuple]. As such the function checks
        such dict instances via recursion.

        Args:
            strategy: The strategy to validate. `BaseStrategy`, tuple,
            or a dict of such

        Returns:
            bool: strategy is | isn't valid

        A valid strategy is one of the following:
            - A `BaseStrategy` instance.
            - A dictionary where each item is a valid strategy.
        """
        if isinstance(strategy, BaseStrategy):
            return True
        return False

    def reset(self):
        """Reset strategy collection to empty default dictionary"""
        self.__init__()


# EXCEPTION CLASS


class StrategyTypeError(Exception):
    """Strategy type not valid errors"""

    def __init__(self, strategy: BaseStrategy | tuple):
        msg = "Input is not valid."  # i.e. default
        if inspect.isclass(strategy):
            msg += f"class must be an instance of `BaseStrategy` not: {type(strategy())}"
        if isinstance(strategy, dict):
            msg += "dict items must be a list of `BaseStrategy` or tuples"
        super().__init__(msg)


# PARTITION PROCESSING:


def _process_partition(
    partition_iter: Iterator[Row],
    strategies: StrategyMapCollection,
    id: str,
    attr: str | None,
    canonicalization_rule: Rule = "first",
) -> Iterator[Row]:
    """process a spark dataframe partition i.e. a list[Row]

    This function is functionality mapped to a worker node. For clean
    separation from the driver, strategies are re-instantiated and the main
    dupegrouper API is executed *per* worker node.

    Args:
        paritition_iter: a partition
        strategies: the collection of strategies
        id: the unique identified of the dataset a.k.a "business key"
        attr: the attribute on which to deduplicate

    Returns:
        A list[Row], deduplicated
    """
    # handle empty partitions
    rows = list(partition_iter)
    if not rows:
        return iter([])

    # re-instantiate strategies based on driver's
    reinstantiated_strategies = {}
    for key, values in strategies.items():
        reinstantiated_strategies[key] = [
            v if isinstance(v, tuple) else v.reinstantiate()
            #
            for v in values
        ]

    # Core API reused per partition, per worker node
    dg = Duped(rows, id=id, canonicalization_rule=canonicalization_rule)
    dg.apply(strategies)
    dg.canonicalize(attr)

    return iter(dg.df)  # type: ignore[arg-type]
