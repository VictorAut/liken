import pyarrow as pa
import pytest

import liken as lk
from liken.exceptions import InvalidArgumentTypeError
from liken.exceptions import InvalidArgumentValueError


BAD_PREPROCESSORS = ["not_a_preprocessor", 123, object()]


# PROCESSORS PROPAGATION:


@pytest.mark.parametrize("preprocessors", [[lk.preprocessors.strip()]])
def test_pipeline_preprocessors_propagate_to_step(preprocessors):
    pipeline = lk.pipeline(preprocessors=preprocessors).step(lk.col("email").exact())

    step = pipeline.steps[0]

    assert all(s.preprocessors == preprocessors for s in step)


@pytest.mark.parametrize("preprocessors", [[lk.preprocessors.strip()]])
def test_pipeline_preprocessors_propagate_to_on(preprocessors):
    pipeline = lk.pipeline(preprocessors=preprocessors).step(lk.col("email").exact())

    step = pipeline.steps[0]

    # each unit corresponds to an `on`
    assert all(s.preprocessors == preprocessors for s in step)


def test_on_preprocessors_override_step_and_pipeline():
    pipeline_pre = [lk.preprocessors.strip()]
    step_pre = [lk.preprocessors.lower()]
    on_pre = [lk.preprocessors.alnum()]

    pipeline = lk.pipeline(preprocessors=pipeline_pre).step(
        lk.col("email", preprocessors=on_pre).exact(),
        preprocessors=step_pre,
    )

    step = pipeline.steps[0]

    assert all(s.preprocessors == on_pre for s in step)


def test_step_preprocessors_override_pipeline():
    pipeline_pre = [lk.preprocessors.strip()]
    step_pre = [lk.preprocessors.lower()]

    pipeline = lk.pipeline(preprocessors=pipeline_pre).step(
        lk.col("email").exact(),
        preprocessors=step_pre,
    )

    step = pipeline.steps[0]

    assert all(s.preprocessors == step_pre for s in step)


def test_preprocessors_only_fill_missing():
    pipeline_pre = [lk.preprocessors.strip()]
    on_pre = [lk.preprocessors.lower()]

    pipeline = lk.pipeline(preprocessors=pipeline_pre).step(
        [
            lk.col("email", preprocessors=on_pre).exact(),
            lk.col("address").exact(),
        ]
    )

    step = pipeline.steps[0]

    # first keeps its own
    assert step[0].preprocessors == on_pre

    # second inherits from pipeline
    assert step[1].preprocessors == pipeline_pre


# BAD PREPROCESSORS


@pytest.mark.parametrize(
    "bad_preprocessor",
    BAD_PREPROCESSORS,
)
def test_pipeline_rejects_invalid_global_preprocessor(bad_preprocessor):
    with pytest.raises(TypeError, match=r"`preprocessor` must be of type .Preprocessor., got"):
        lk.pipeline(preprocessors=[bad_preprocessor]).step(lk.col("email").exact())


@pytest.mark.parametrize(
    "bad_preprocessor",
    BAD_PREPROCESSORS,
)
def test_pipeline_rejects_invalid_step_preprocessor(bad_preprocessor):
    with pytest.raises(TypeError, match=r"`preprocessor` must be of type .Preprocessor., got"):
        lk.pipeline().step(
            lk.col("email").exact(),
            preprocessors=[bad_preprocessor],
        )


@pytest.mark.parametrize(
    "bad_preprocessor",
    BAD_PREPROCESSORS,
)
def test_pipeline_rejects_invalid_on_preprocessor(bad_preprocessor):
    with pytest.raises(TypeError, match=r"`preprocessor` must be of type .Preprocessor., got"):
        lk.pipeline().step(lk.col("email", preprocessors=[bad_preprocessor]).exact())


# COLLAPSE WHITESPACE:


def test_collapse_whitespace_collapses_internal_runs():
    pp = lk.preprocessors.collapse_whitespace()
    pp.from_array(pa.array(["quick  brown\tfox"]))

    assert pp.process().to_pylist() == ["quick brown fox"]


def test_collapse_whitespace_collapses_tabs_and_newlines():
    pp = lk.preprocessors.collapse_whitespace()
    pp.from_array(pa.array(["a\tb\nc\rd"]))

    assert pp.process().to_pylist() == ["a b c d"]


def test_collapse_whitespace_edge_runs_become_single_space():
    pp = lk.preprocessors.collapse_whitespace()
    pp.from_array(pa.array(["  quick  brown  "]))

    assert pp.process().to_pylist() == [" quick brown "]


def test_collapse_whitespace_propagates_nulls():
    pp = lk.preprocessors.collapse_whitespace()
    pp.from_array(pa.array(["a  b", None]))

    assert pp.process().to_pylist() == ["a b", None]


def test_collapse_whitespace_keeps_empty_strings():
    pp = lk.preprocessors.collapse_whitespace()
    pp.from_array(pa.array([""]))

    assert pp.process().to_pylist() == [""]


def test_collapse_whitespace_leaves_non_ascii_whitespace():
    pp = lk.preprocessors.collapse_whitespace()
    pp.from_array(pa.array(["a\x0bb", "a\xa0b"]))

    assert pp.process().to_pylist() == ["a\x0bb", "a\xa0b"]


# REGEX REPLACE:


def test_regex_replace_literal_match():
    pp = lk.preprocessors.regex_replace(",", " ")
    pp.from_array(pa.array(["123ab,OL5"]))

    assert pp.process().to_pylist() == ["123ab OL5"]


def test_regex_replace_group_pattern():
    pp = lk.preprocessors.regex_replace(r"(ol)", "xx")
    pp.from_array(pa.array(["holey molley"]))

    assert pp.process().to_pylist() == ["hxxey mxxley"]


def test_regex_replace_propagates_nulls():
    pp = lk.preprocessors.regex_replace(",", " ")
    pp.from_array(pa.array(["a,b", None]))

    assert pp.process().to_pylist() == ["a b", None]


def test_regex_replace_invalid_pattern_raises_at_construction():
    """U27: o padrão inválido é rejeitado já na construção, não no process."""
    with pytest.raises(InvalidArgumentValueError, match="`pattern` must be a valid regular expression"):
        lk.preprocessors.regex_replace("[", "")


######################
# eager validation  #
#####################


def test_normalize_unicode_rejects_unknown_form():
    """U27: an unknown normalization form is rejected at construction."""
    with pytest.raises(InvalidArgumentValueError, match="`form` must be one of"):
        lk.preprocessors.normalize_unicode(form="NOTAFORM")


def test_regex_replace_rejects_invalid_pattern():
    with pytest.raises(InvalidArgumentValueError, match="`pattern` must be a valid regular expression"):
        lk.preprocessors.regex_replace("[[", "x")


def test_regex_replace_rejects_non_str_replacement():
    with pytest.raises(InvalidArgumentTypeError, match="`replacement` must be of type 'str', got 'int'"):
        lk.preprocessors.regex_replace("a", 1)


def test_remove_stopwords_rejects_unknown_language():
    with pytest.raises(InvalidArgumentValueError, match="`language` must be one of"):
        lk.preprocessors.remove_stopwords(language="klingon")


def test_remove_stopwords_rejects_empty_words():
    """U42: `[]` is rejected; `None` means the language defaults."""
    with pytest.raises(InvalidArgumentValueError, match="`words` must be a non-empty list"):
        lk.preprocessors.remove_stopwords(words=[])


def test_remove_stopwords_rejects_non_str_words():
    with pytest.raises(InvalidArgumentTypeError, match=r"`words\[0\]` must be of type 'str', got 'int'"):
        lk.preprocessors.remove_stopwords(words=[1])


def test_remove_stopwords_process_does_not_mutate_words():
    """U42: `process` must not overwrite the stored words."""
    preprocessor = lk.preprocessors.remove_stopwords(words=["the"])
    preprocessor.from_array(pa.array(["the cat"]))
    preprocessor.process()

    assert preprocessor._words == ("the",)


def test_ascii_fold_takes_no_form_argument():
    """U53: the dead `form` parameter is removed; the filter hardcodes NFKD."""
    from liken.preprocessors import AsciiFold

    with pytest.raises(TypeError):
        AsciiFold(form="NFC")


####################
# repr and str    #
####################


def test_preprocessor_repr_records_constructor_args():
    """U45: preprocessors render their constructor arguments."""
    assert repr(lk.preprocessors.lower()) == "Lower()"
    assert repr(lk.preprocessors.normalize_unicode()) == "NormalizeUnicode(form='NFKD')"
    assert repr(lk.preprocessors.regex_replace("a", "b")) == "RegexReplace('a', 'b')"
    assert (
        repr(lk.preprocessors.remove_stopwords(words=["the"])) == "RemoveStopwords(words=('the',), language='english')"
    )


def test_col_str_includes_step_preprocessors():
    """U45: a Col's string representation includes its preprocessors."""
    c = lk.col("a", preprocessors=[lk.preprocessors.lower()]).exact()

    assert "preprocessors=['Lower()']" in str(c)
