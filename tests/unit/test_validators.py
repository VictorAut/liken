import pytest

from liken.core.deduper import BaseDeduper
from liken.core.deduper import ThresholdDeduper
from liken.dedupers.exact import exact
from liken.dedupers.fuzzy import fuzzy
from liken.dedupers.jaccard import jaccard
from liken.validators import validate_deduper_arg
from liken.validators import validate_explore_column_exists
from liken.validators import validate_explore_columns_arg
from liken.validators import validate_frac_arg
from liken.validators import validate_thresholds_arg


# validate_deduper_arg


def test_validate_deduper_arg_accepts_base_deduper():
    deduper = exact()
    assert validate_deduper_arg(deduper) is deduper


def test_validate_deduper_arg_rejects_non_deduper():
    with pytest.raises(TypeError, match="deduper must be instance of BaseDeduper, got str"):
        validate_deduper_arg("exact()")


# validate_frac_arg


@pytest.mark.parametrize("frac", [0.5, 1.0])
def test_validate_frac_arg_accepts_valid(frac):
    assert validate_frac_arg(frac) == frac


@pytest.mark.parametrize("frac", ["0.5", None, True, 0, -0.5, 1.5])
def test_validate_frac_arg_rejects_invalid(frac):
    with pytest.raises(ValueError, match="frac must be a number in the range"):
        validate_frac_arg(frac)


# validate_thresholds_arg


def test_validate_thresholds_arg_accepts_valid():
    assert validate_thresholds_arg([0.5, 0.9]) == [0.5, 0.9]


@pytest.mark.parametrize("thresholds", ["0.5", (), []])
def test_validate_thresholds_arg_rejects_non_list_or_empty(thresholds):
    with pytest.raises(ValueError, match="thresholds must be a non-empty list"):
        validate_thresholds_arg(thresholds)


@pytest.mark.parametrize("thresholds", [[0], [1], [1.5], [-0.1], [True], [0.5, "0.9"]])
def test_validate_thresholds_arg_rejects_invalid_members(thresholds):
    with pytest.raises(ValueError, match="thresholds must be a non-empty list"):
        validate_thresholds_arg(thresholds)


# validate_explore_columns_arg


def test_validate_explore_columns_arg_accepts_list():
    assert validate_explore_columns_arg(["address", "email"]) == ["address", "email"]


def test_validate_explore_columns_arg_accepts_dict_of_threshold_dedupers():
    columns = {"address": fuzzy()}
    assert validate_explore_columns_arg(columns) == columns


@pytest.mark.parametrize("columns", [[], ["address", 1]])
def test_validate_explore_columns_arg_rejects_invalid_list(columns):
    with pytest.raises(ValueError, match="columns for explore must be a non-empty list"):
        validate_explore_columns_arg(columns)


@pytest.mark.parametrize("columns", [{}, {1: fuzzy()}])
def test_validate_explore_columns_arg_rejects_invalid_dict(columns):
    with pytest.raises(ValueError, match="columns for explore must be a non-empty list"):
        validate_explore_columns_arg(columns)


@pytest.mark.parametrize(
    "columns",
    [
        {"address": exact()},  # predicate deduper, not a threshold deduper
        {"address": jaccard()},  # compound-column deduper
    ],
)
def test_validate_explore_columns_arg_rejects_invalid_deduper(columns):
    with pytest.raises(ValueError, match="explore dedupers must be single-column similarity"):
        validate_explore_columns_arg(columns)


def test_validate_explore_columns_arg_rejects_other_types():
    with pytest.raises(ValueError, match="columns for explore must be a non-empty list"):
        validate_explore_columns_arg("address")


# validate_explore_column_exists


def test_validate_explore_column_exists_accepts_present_column():
    assert validate_explore_column_exists("address", ["id", "address"]) == "address"


def test_validate_explore_column_exists_rejects_missing_column():
    with pytest.raises(ValueError, match="column 'address' not found in the dataframe"):
        validate_explore_column_exists("address", ["id", "email"])


# ThresholdDeduper bounds


@pytest.mark.parametrize("threshold", [0, 0.5, 0.99])
def test_threshold_deduper_accepts_threshold_below_one(threshold):
    assert isinstance(ThresholdDeduper(threshold=threshold), BaseDeduper)


@pytest.mark.parametrize("threshold", [-0.1, 1.0, 1.5])
def test_threshold_deduper_rejects_threshold_outside_bounds(threshold):
    with pytest.raises(ValueError, match="threshold value must be greater or equal to 0 and less than 1"):
        ThresholdDeduper(threshold=threshold)
