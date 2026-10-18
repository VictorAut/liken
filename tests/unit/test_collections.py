import pytest

import liken as lk
from liken.collections.base import CollectionsManager
from liken.collections.base import InvalidDeduperError
from liken.collections.dict import DeduplicationDict
from liken.collections.pipelines import Col
from liken.collections.pipelines import Pipeline
from liken.collections.pipelines import col
from liken.constants import SEQUENTIAL_API_DEFAULT_KEY
from liken.core.deduper import BaseDeduper
from liken.core.registries import dedupers_registry
from liken.exceptions import InvalidArgumentTypeError
from liken.exceptions import InvalidArgumentValueError
from liken.preprocessors import Lower
from liken.preprocessors import lower


###########
# Helpers #
###########


class DummyDeduper(BaseDeduper):
    def __str__(self):
        return self.str_representation("dummy_deduper")


@pytest.fixture
def s1():
    return DummyDeduper()


@pytest.fixture
def s2():
    return DummyDeduper()


@pytest.fixture
def s3():
    return DummyDeduper()


# The registry factories construct fresh instances; the `s1`/`s2`/`s3`
# fixtures provide distinct instances for identity assertions.
dedupers_registry.register("s1", func=DummyDeduper)
dedupers_registry.register("s2", func=DummyDeduper)
dedupers_registry.register("s3", func=DummyDeduper)

#####################
# DeduplicationDict tests
#####################


@pytest.mark.parametrize(
    "columns, deduper",
    [
        ("address", [BaseDeduper()]),
        ("address", (BaseDeduper(),)),
        (("address", "email"), [BaseDeduper()]),
        (("address", "email"), (BaseDeduper(),)),
    ],
)
def test_dedupers_config_accepts_inputs(columns, deduper):
    config = DeduplicationDict()
    config[columns] = deduper

    assert columns in config
    assert config[columns] == deduper


def test_dedupers_config_rejects_invalid_key_type(s1):
    config = DeduplicationDict()
    with pytest.raises(InvalidDeduperError, match="Invalid type for dict key type"):
        config[123] = [s1]


def test_dedupers_config_rejects_invalid_value_type():
    config = DeduplicationDict()
    with pytest.raises(InvalidDeduperError, match="Invalid type for dict value"):
        config["col"] = "not_a_deduper"


def test_dedupers_config_rejects_invalid_member_in_value(s1, s2, s3):
    config = DeduplicationDict()
    with pytest.raises(
        InvalidDeduperError,
        match=r"Invalid type for dict value member: at index 1 for key 'col': expected 'BaseDeduper', got 'str'",
    ):
        config["col"] = [s1, "bad", s2, s3]


def test_dict_api_rejects_preprocessor_with_explicit_message():
    """U44/D3: a dict API permanece sem preprocessors e a mensagem os nomeia."""
    sm = CollectionsManager()
    with pytest.raises(InvalidDeduperError, match="preprocessors are not accepted in the dict API"):
        sm.apply({"a": lower()})


################
# apply method #
################


def test_collections_manager_apply_sequential_once(s1):
    sm = CollectionsManager()
    sm.apply(s1)

    dedupers = sm.get()
    assert s1 in dedupers[SEQUENTIAL_API_DEFAULT_KEY]


def test_collections_manager_apply_sequential_multiple(s1, s2, s3):
    sm = CollectionsManager()
    sm.apply(s1)
    sm.apply(s2)
    sm.apply(s3)

    dedupers = sm.get()
    assert s1 in dedupers[SEQUENTIAL_API_DEFAULT_KEY]
    assert s2 in dedupers[SEQUENTIAL_API_DEFAULT_KEY]
    assert s3 in dedupers[SEQUENTIAL_API_DEFAULT_KEY]


def test_collections_manager_apply_dict_single(s1, s2, s3):
    sm = CollectionsManager()
    dedupe_dict = {"a": [s1], "b": (s2), "c": s3}

    sm.apply(dedupe_dict)
    result = sm.get()

    assert result["a"] == [s1]
    assert result["b"] == (s2,)
    assert result["c"] == (s3,)


def test_collections_manager_apply_dict(s1, s2, s3):
    sm = CollectionsManager()
    dedupe_dict = {"a": [s1], "b": (s2, s3)}

    sm.apply(dedupe_dict)
    result = sm.get()

    assert result["a"] == [s1]
    assert result["b"] == (s2, s3)


def test_collections_manager_apply_single_on_no_pipeline(s1):
    sm = CollectionsManager()
    on_deduper = col("a").s1()

    sm.apply(on_deduper)
    result = sm.get()

    assert isinstance(result, Pipeline)
    dedupers = result.steps[0]
    assert len(dedupers) == 1

    assert isinstance(dedupers[0], tuple)
    assert dedupers[0][0] == "a"


def test_collections_manager_apply_single_on_as_pipeline(s1):
    sm = CollectionsManager()
    pipeline = Pipeline().step(col("a").s1())

    sm.apply(pipeline)
    result = sm.get()

    assert isinstance(result, Pipeline)
    dedupers = result.steps[0]
    assert len(dedupers) == 1

    assert isinstance(dedupers[0], tuple)
    assert dedupers[0][0] == "a"


def test_collections_manager_apply_stepped_pipeline(s1, s2, s3):
    sm = CollectionsManager()
    pipeline = Pipeline().step(col("a").s1()).step([col("b").s2(), col("c").s3()])

    sm.apply(pipeline)
    result = sm.get()

    assert isinstance(result, Pipeline)
    dedupers = result.steps
    assert len(dedupers) == 2

    print(dedupers)

    assert isinstance(dedupers[0][0], tuple)
    assert dedupers[0][0][0] == "a"
    assert isinstance(dedupers[1][0], tuple)
    assert dedupers[1][0][0] == "b"
    assert isinstance(dedupers[1][1], tuple)
    assert dedupers[1][1][0] == "c"


def test_collections_manager_apply_rejects_invalid_type():
    sm = CollectionsManager()
    with pytest.raises(InvalidDeduperError, match="Invalid deduper"):
        sm.apply(123)


def test_collections_manager_apply_rejects_empty_dict():
    sm = CollectionsManager()
    with pytest.raises(
        InvalidDeduperError,
        match="expected a non-empty dict of column labels to dedupers, got an empty dict",
    ):
        sm.apply({})


def test_collections_manager_apply_rejects_empty_sequence_value(s1):
    sm = CollectionsManager()
    with pytest.raises(
        InvalidDeduperError,
        match="key 'a' maps to an empty sequence; expected at least one deduper",
    ):
        sm.apply({"a": []})


def test_collections_manager_apply_rejects_zero_step_pipeline():
    sm = CollectionsManager()
    with pytest.raises(InvalidDeduperError, match="at least one step"):
        sm.apply(lk.pipeline())


def test_collections_manager_apply_rejects_reserved_default_key(s1):
    sm = CollectionsManager()
    with pytest.raises(InvalidDeduperError, match="reserved"):
        sm.apply({SEQUENTIAL_API_DEFAULT_KEY: [s1]})


def test_collections_manager_apply_rejection_does_not_set_has_applies():
    """A rejected apply does not mark `has_applies`, so the exact fallback
    continues to trigger on the next enact."""
    sm = CollectionsManager()
    with pytest.raises(InvalidDeduperError):
        sm.apply({})

    assert sm.has_applies is False


def test_collections_manager_apply_rejects_sequence_after_dict(s1, s2, s3):
    sm = CollectionsManager()
    sm.apply({"a": (s1,), "b": (s1, s2)})  # legal
    with pytest.raises(
        InvalidDeduperError,
        match="Cannot apply a 'BaseDeduper' after a deduper mapping",
    ):
        sm.apply(s3)


def test_collections_manager_apply_warns_dict_after_sequence():
    sm = CollectionsManager()
    deduper = BaseDeduper()

    sm.apply(deduper)

    with pytest.warns(
        UserWarning,
        match="Replacing previously added sequence deduper with a dict deduper",
    ):
        sm.apply({"email": [deduper]})


def test_collections_manager_apply_dict_after_pipeline(s1):
    sm = CollectionsManager()
    sm.apply(Pipeline().step(col("a").s1()))

    with pytest.warns(UserWarning, match="Replacing previously added 'Pipeline' deduper with a dict deduper"):
        sm.apply({"email": (s1,)})

    result = sm.get()
    assert isinstance(result, DeduplicationDict)


##################
# Replace policy #
##################


def test_collections_manager_apply_warns_dict_after_dict(s1, s2):
    """dict-after-dict warns ONLY and replaces"""
    sm = CollectionsManager()
    sm.apply({"a": (s1,)})

    with pytest.warns(UserWarning, match="Replacing previously added dict deduper with a new dict deduper"):
        sm.apply({"b": (s2,)})

    result = sm.get()
    assert isinstance(result, DeduplicationDict)
    assert "b" in result
    assert "a" not in result


def test_collections_manager_apply_warns_pipeline_after_sequence(s1):
    sm = CollectionsManager()
    sm.apply(s1)

    with pytest.warns(UserWarning, match="Replacing previously added sequence deduper with a 'Pipeline' deduper"):
        sm.apply(Pipeline().step(col("a").s1()))


def test_collections_manager_apply_warns_pipeline_after_dict(s1):
    sm = CollectionsManager()
    sm.apply({"a": (s1,)})

    with pytest.warns(UserWarning, match="Replacing previously added dict deduper with a 'Pipeline' deduper"):
        sm.apply(Pipeline().step(col("a").s1()))


def test_collections_manager_apply_warns_dict_after_pipeline(s1):
    sm = CollectionsManager()
    sm.apply(Pipeline().step(col("a").s1()))

    with pytest.warns(UserWarning, match="Replacing previously added 'Pipeline' deduper with a dict deduper"):
        sm.apply({"a": (s1,)})

    assert isinstance(sm.get(), DeduplicationDict)


def test_collections_manager_apply_rejects_sequence_after_pipeline(s1):
    """the rejection names the actual current API mode, here 'Pipeline'."""
    sm = CollectionsManager()
    sm.apply(Pipeline().step(col("a").s1()))

    with pytest.raises(
        InvalidDeduperError,
        match="Cannot apply a 'BaseDeduper' after a 'Pipeline' deduper has been set",
    ):
        sm.apply(s1)


def test_collections_manager_apply_rejects_sequence_after_dict_names_dict(s1):
    """the rejection names the actual current API mode, here the dict."""
    sm = CollectionsManager()
    sm.apply({"a": (s1,)})

    with pytest.raises(
        InvalidDeduperError,
        match="Cannot apply a 'BaseDeduper' after a deduper mapping \\(dict\\) has been set",
    ):
        sm.apply(s1)


#######
# get #
#######


def test_collections_manager_get_returns_config():
    sm = CollectionsManager()
    result = sm.get()
    assert isinstance(result, DeduplicationDict)


################
# mode checks  #
################


def test_collections_manager_validate_api_mode_sequential_requires_columns(s1):
    """sequential mode — including the exact fallback — requires
    `columns`; the mode rule lives on the collections manager."""
    sm = CollectionsManager()
    sm.apply(s1)

    with pytest.raises(InvalidArgumentValueError, match="columns cannot be None"):
        sm.validate_api_mode(None)
    with pytest.raises(InvalidArgumentValueError, match="columns cannot be None"):
        sm.validate_api_mode(())


def test_collections_manager_validate_api_mode_sequential_accepts_columns(s1):
    sm = CollectionsManager()
    sm.apply(s1)

    assert sm.validate_api_mode("a") == "a"
    assert sm.validate_api_mode(("a", "b")) == ("a", "b")


def test_collections_manager_validate_api_mode_dict_requires_none(s1):
    sm = CollectionsManager()
    sm.apply({"a": (s1,)})

    with pytest.raises(InvalidArgumentValueError, match="columns must be None when using the dict API"):
        sm.validate_api_mode("a")
    assert sm.validate_api_mode(None) is None


def test_collections_manager_validate_api_mode_pipeline_requires_none(s1):
    sm = CollectionsManager()
    sm.apply(Pipeline().step(col("a").s1()))

    with pytest.raises(InvalidArgumentValueError, match="columns must be None when using the dict API"):
        sm.validate_api_mode("a")
    assert sm.validate_api_mode(None) is None


##############
# pretty_get #
##############


def test_pretty_get_sequential_api():
    sm = CollectionsManager()
    sm.apply(lk.fuzzy())
    assert sm.pretty_get() == "fuzzy(threshold=0.95, scorer='simple_ratio')"


def test_pretty_get_dict_api(s1, s2, s3):
    sm = CollectionsManager()
    sm.apply({"col_a": [s1, s3], "col_b": [s2]})

    pretty = sm.pretty_get()
    assert (
        pretty == "{"
        "\n\t'col_a': ("
        "\n\t\tdummy_deduper(),"
        "\n\t\tdummy_deduper(),"
        "\n\t\t),"
        "\n\t'col_b': ("
        "\n\t\tdummy_deduper(),"
        "\n\t\t),"
        "\n}"
    )


def test_pretty_get_rules_api():
    sm = CollectionsManager()
    sm.apply(Pipeline().step(col("col_a").exact()).step(col("col_b").fuzzy()))

    pretty = sm.pretty_get()
    assert (
        pretty == "("
        "\n\tlk.pipeline()"
        "\n\t\t.step(["
        "\n\t\t\tlk.col('col_a').exact(),"
        "\n\t\t])"
        "\n\t\t.step(["
        "\n\t\t\tlk.col('col_b').fuzzy(threshold=0.95, scorer='simple_ratio'),"
        "\n\t\t])"
        "\n)"
    )


#########
# reset #
#########


def test_collections_manager_reset_clears_collection(s1):
    sm = CollectionsManager()

    assert sm.get() == {SEQUENTIAL_API_DEFAULT_KEY: []}
    assert sm.pretty_get() is None

    sm.apply(s1)
    sm.reset()

    assert sm.get() == {SEQUENTIAL_API_DEFAULT_KEY: []}
    assert sm.pretty_get() is None


def test_collections_manager_reset_resets_has_applies(s1):
    """after a reset the exact fallback must be able to fire again."""
    sm = CollectionsManager()
    sm.apply(s1)
    assert sm.has_applies

    sm.reset()

    assert not sm.has_applies


###########################
# InvalidDeduperError #
###########################


def test_deduper_config_type_error():
    err = InvalidDeduperError("bad")
    assert isinstance(err, TypeError)


##################
# Pipeline misuse #
##################


def test_invert_rejects_non_predicate_deduper():
    with pytest.raises(TypeError, match="`deduper` must be of type 'PredicateDeduper'"):
        ~lk.col("address").exact()


def test_step_rejects_non_col():
    with pytest.raises(TypeError, match="`cols` must be of type 'Col'"):
        lk.pipeline().step("address")


#######################
# step argument checks #
#######################


def test_step_rejects_empty_list():
    """an empty pipeline step list broke execution with an UnboundLocalError."""
    with pytest.raises(InvalidArgumentValueError, match="`cols` must be a non-empty list"):
        lk.pipeline().step([])


def test_step_rejects_non_col_list_member():
    """every list member is validated, matching the single-arg case."""
    with pytest.raises(InvalidArgumentTypeError, match="`cols` must be of type 'Col', got 'str'"):
        lk.pipeline().step(["a", "b"])


def test_step_preprocessors_omitted_inherits_pipeline():
    """an omitted `step` preprocessors arg inherits the pipeline's."""
    p = lk.pipeline(preprocessors=[lower()]).step(lk.col("a").fuzzy())

    assert len(p.steps[0][0].preprocessors) == 1


def test_step_preprocessors_empty_list_means_none():
    """`[]` means no preprocessors for the step — no inheritance."""
    p = lk.pipeline(preprocessors=[lower()]).step(lk.col("a").fuzzy(), preprocessors=[])

    assert p.steps[0][0].preprocessors == []


def test_step_preprocessors_none_means_none():
    p = lk.pipeline(preprocessors=[lower()]).step(lk.col("a").fuzzy(), preprocessors=None)

    assert p.steps[0][0].preprocessors == []


##############
# Col guards #
##############


def test_col_rechaining_warns():
    """chaining a second deduper on one Col warns; last-wins is kept."""
    with pytest.warns(UserWarning, match="replaces the previously chained deduper"):
        lk.col("a").exact().fuzzy()


def test_col_reuse_across_steps_warns():
    c = lk.col("a").exact()
    p = lk.pipeline()
    p.step(c)

    with pytest.warns(UserWarning, match="more than one step"):
        p.step(c)


def test_col_unit_before_chaining_raises():
    """an unset `_unit` raises a liken error, not the registry's."""
    with pytest.raises(InvalidDeduperError, match="no deduper chained yet"):
        _ = lk.col("a").unit
    with pytest.raises(InvalidDeduperError, match="no deduper chained yet"):
        _ = lk.col("a")._unit


def test_single_preprocessor_is_resolved_to_list():
    c = lk.col("address", preprocessors=lower()).fuzzy()

    assert len(c.unit.preprocessors) == 1
    assert isinstance(c.unit.preprocessors[0], Lower)


def test_col_isin_method_wrapper():
    c = lk.col("address").isin(values=["london"])

    assert c.unit.columns == "address"
    assert str(c.unit.deduper).startswith("isin(")


def test_col_str_negated():
    c = ~lk.col("address").isna()

    assert str(c) == "~lk.col('address').isna()"


def test_apply_pipeline_twice_warns():
    manager = CollectionsManager()
    manager.apply(lk.pipeline().step(lk.col("address").exact()))

    with pytest.warns(UserWarning, match="Replacing previously added 'Pipeline' deduper"):
        manager.apply(lk.pipeline().step(lk.col("email").exact()))


def test_pipeline_str_includes_preprocessors():
    from liken.preprocessors import lower

    pipeline = lk.pipeline(preprocessors=[lower()]).step(lk.col("address").exact())

    representation = str(pipeline)
    assert "preprocessors=" in representation
    assert "lk.col('address').exact()" in representation


def test_pipeline_str_emits_pasteable_pipeline():
    p = lk.pipeline().step(lk.col("email").fuzzy())

    s = str(p)

    assert "lk.pipeline" in s
    assert "lk.col" in s
    assert "lk.rules" not in s


###########################
# Col deduper stubs guard #
###########################


BUILTIN_DEDUPER_NAMES = [
    "exact",
    "fuzzy",
    "tfidf",
    "lsh",
    "jaccard",
    "cosine",
    "edit_distance",
    "isin",
    "isna",
    "str_contains",
    "str_endswith",
    "str_len",
    "str_startswith",
]


@pytest.mark.parametrize("name", BUILTIN_DEDUPER_NAMES)
def test_col_has_stub_method_for_every_builtin_deduper(name):
    # A new deduper registered without a Col stub fails here.
    assert hasattr(Col, name)
