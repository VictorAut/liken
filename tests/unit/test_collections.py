import pytest

import liken as lk
from liken._collections import SEQUENTIAL_API_DEFAULT_KEY
from liken._collections import CollectionsManager
from liken._collections import DeduplicationDict
from liken._collections import InvalidDeduperError
from liken._dedupers import BaseDeduper
from liken._pipelines import Pipeline
from liken._pipelines import col
from liken.core.registries import dedupers_registry


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


dedupers_registry.register("s1", func=lambda: s1)
dedupers_registry.register("s2", func=lambda: s2)
dedupers_registry.register("s3", func=lambda: s3)

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
    with pytest.raises(InvalidDeduperError, match="Invalid type for dict value member"):
        config["col"] = [s1, "bad", s2, s3]


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


#######
# get #
#######


def test_collections_manager_get_returns_config():
    sm = CollectionsManager()
    result = sm.get()
    assert isinstance(result, DeduplicationDict)


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
        "\n\tlk.rules.builder()"
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


###########################
# InvalidDeduperError #
###########################


def test_deduper_config_type_error():
    err = InvalidDeduperError("bad")
    assert isinstance(err, TypeError)
