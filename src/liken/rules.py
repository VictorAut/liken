from liken._rules import On
from liken._rules import Pipeline
from liken._processors import Processor
from liken._types import Columns


# PUBLIC ON API:

def pipeline(processors: Processor | list[Processor] = []) -> Pipeline:
    """TODO"""
    return Pipeline(processors)

# def pipeline(*dedupers) -> Pipeline:
#     """TODO"""
#     return Pipeline(*dedupers)

def on(columns: Columns, /) -> On:
    """Unit container for a single strategy in the Pipeline API.

    Operates a "strat" on a "columns". Is provided as comma separated members to
    `Pipeline`. Allows for "and" chaining via the `&` operator to logically
    compose strategy "rules".

    The `&` ("and") operator is the only supplier logical combination operator
    supplier, as the equivalent to "or" is achieved by comma separating `on`
    calls inside `Pipeline`. The results of `&` are interepreted as boolean and
    whereby the left-hand deduplication strategy must agree with the right-hand
    strategy for any given pairwise combination.

    Args:
        columns: the label(s) of a column or columns.
        strat: the strategy to apply.

    Returns:
        None

    Example:
        single ``on`` strategy:

            from liken import Dedupe, exact, fuzzy
            from liken.rules import Pipeline, on, isna, str_endswith

            on("address", exact())

        Strategies combined with ``&``:

            on("email", fuzzy(threshold=0.95)) & on("email", str_endswith("UK"))

        Strategies can be combined with ``&`` for **different** columns:

            on("email", fuzzy(threshold=0.95)) & on("address", ~isna())

        The above can be read as "deduplicate email only when the address field
        is not null":

            >>> df # Before
            +------+-----------+---------------------+
            | id   |  address  |        email        |
            +------+-----------+---------------------+
            |  1   |  london   |  foobar@gmail.com   |
            |  2   |   paris   |  Foobar@gmail.com   |
            |  3   |   null    |  fooBar@gmail.com   |
            +------+-----------+---------------------+

            >>> df # After
            +------+-----------+---------------------+--------------+
            | id   |  address  |        email        | canonical_id |
            +------+-----------+---------------------+--------------+
            |  1   |  london   |  foobar@gmail.com   |       1      |
            |  2   |   paris   |  Foobar@gmail.com   |       1      |
            |  3   |   null    |  fooBar@gmail.com   |       3      |
            +------+-----------+---------------------+--------------+

        Where the first two rows are now linked via the same canonical_id.
    """
    return On(columns)
