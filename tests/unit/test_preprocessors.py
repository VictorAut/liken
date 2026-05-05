import pytest

import liken as lk


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
    with pytest.raises(TypeError, match="Invalid arg: preprocessor must be instance of Preprocessor"):
        lk.pipeline(preprocessors=[bad_preprocessor]).step(lk.col("email").exact())


@pytest.mark.parametrize(
    "bad_preprocessor",
    BAD_PREPROCESSORS,
)
def test_pipeline_rejects_invalid_step_preprocessor(bad_preprocessor):
    with pytest.raises(TypeError, match="Invalid arg: preprocessor must be instance of Preprocessor"):
        lk.pipeline().step(
            lk.col("email").exact(),
            preprocessors=[bad_preprocessor],
        )


@pytest.mark.parametrize(
    "bad_preprocessor",
    BAD_PREPROCESSORS,
)
def test_pipeline_rejects_invalid_on_preprocessor(bad_preprocessor):
    with pytest.raises(TypeError, match="Invalid arg: preprocessor must be instance of Preprocessor"):
        lk.pipeline().step(lk.col("email", preprocessors=[bad_preprocessor]).exact())
