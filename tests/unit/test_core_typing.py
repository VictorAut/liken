"""Type-contract tests for core module signatures."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from collections.abc import Iterable
from typing import get_args

from liken.backends.pandas.affordances import register_pd_affordances
from liken.collections.dict import DeduplicationDict
from liken.collections.pipelines import Col
from liken.core.backend import Backend
from liken.core.deduper import BaseDeduper
from liken.core.deduper import PredicateDeduper
from liken.core.deduper import _NegatedPredicateDeduper
from liken.core.wrapper import DF
from liken.core.wrapper import AddsCanonical
from liken.custom import PairGenerator
from liken.custom import register
from liken.datasets import fake_row
from liken.datasets import maybe_null


empty = inspect.Parameter.empty


def _function_annotated(func: Callable[..., object]) -> bool:
    sig = inspect.signature(func)
    return sig.return_annotation is not empty and all(p.annotation is not empty for p in sig.parameters.values())


def _method_annotated(method: Callable[..., object]) -> bool:
    sig = inspect.signature(method)
    params = list(sig.parameters.values())[1:]  # drop `self`
    return sig.return_annotation is not empty and all(p.annotation is not empty for p in params)


def test_backend_protocol_members_are_annotated():
    assert "name" in Backend.__annotations__
    for member in ("is_match", "create_df", "executor", "wrap"):
        assert _method_annotated(getattr(Backend, member)), member


def test_adds_canonical_methods_are_annotated():
    for member in ("_df_as_is", "_df_overwrite_id", "_df_copy_id", "_df_autoincrement_id", "_column_labels_list"):
        assert _method_annotated(getattr(AddsCanonical, member)), member


def test_df_docstring_has_no_unresolved_todo():
    assert "TODO" not in (DF.__doc__ or "")


def test_deduplication_dict_is_generified():
    key_t, value_t = get_args(DeduplicationDict.__orig_bases__[0])
    assert key_t == str | tuple[str, ...]
    assert value_t == list[BaseDeduper] | tuple[BaseDeduper, ...]


def test_col_getattr_and_closure_are_annotated():
    assert _method_annotated(Col.__getattr__)
    wrapper = Col("col").__getattr__("fuzzy")
    assert _function_annotated(wrapper)


def test_predicate_deduper_matches_is_annotated():
    assert _method_annotated(PredicateDeduper._matches)
    assert _method_annotated(_NegatedPredicateDeduper._matches)


def test_public_modules_are_annotated():
    assert _function_annotated(maybe_null)
    assert _function_annotated(fake_row)
    assert _function_annotated(register_pd_affordances)
    assert _function_annotated(register)


def test_pair_generator_element_type_is_object():
    input_t = get_args(PairGenerator)[0][0]
    assert input_t == Iterable[object]
