---
title: Applying Dedupers
---


In the [First Steps](../tutorials/first-steps.md) you found out how to instantiate Liken, and how to explore near duplicate rates in your data. Now, you can *apply* duplicates for execution, by using the `apply` function:

```python
import liken as lk

df = (
    lk.dedupe(df)
    .apply(lk.fuzzy())
    .drop_duplicates("address")
)
```

There are several ways to apply dedupers in Liken. Each offers incremental value, at the cost of complexity. For a quick reference use the following table as a guide to this tutorial:

Different collections of dedupers, whether a single deduper, a dictionary or a pipeline, are best suited to different use cases:

|  | Quick tasks | Multiple columns | Logical combination semantics | Preprocessors |
| ---------- | ---------- | ---------- | ---------- | ---------- |
| [Single Deduper](#single-dedupers) | :white_check_mark: | :material-close: | :material-close: | :material-close: |
| [Dict of dedupers](#dictionaries-of-dedupers) | :white_check_mark: | :white_check_mark: | :material-close: | :material-close: |
| [Pipelines of dedupers](#pipelines-of-dedupers) | :material-close: | :white_check_mark: | :white_check_mark: | :white_check_mark: |


## Single Dedupers

If you only need a single deduper, use it straight in an `apply` function as seen above, i.e. `apply(lk.fuzzy())`, or any other deduper. The column or columns to deduplicate are passed in `drop_duplicates`:

```python
import liken as lk

df = (
    lk.dedupe(df)
    .apply(lk.fuzzy())
    .drop_duplicates("address")
)
```

For a deduper that accepts multiple columns, and keeping only the last instance (not the default first instance):

```python
import liken as lk

df = (
    lk.dedupe(df)
    .apply(lk.cosine())
    .drop_duplicates(("height", "width"), keep = "last")
)
```
!!! info "`keep` arg"
    The `keep` argument accepts the literals "first" or "last" which picks the *representative* of each duplicate cluster.

## Dictionaries of Dedupers

Dictionaries of dedupers have three properties:

- **Keys are column sets.** A key is a column label (`"email"`) or a tuple of column labels (`("first_name", "last_name")`).
- **Values are a deduper or a sequence of dedupers.** A value is one deduper, or a tuple of dedupers chained on that key's columns: a record counts as a duplicate if it is linked by any deduper in the tuple.
- **Keys combine as OR.** Records linked by *any* key's deduper end up in the same cluster.

```python
import liken as lk

collection = {
    "email": lk.exact(),
    "address": (
        lk.fuzzy(threshold=0.98),
        lk.tfidf(threshold=0.9, ngram=(1, 2), topn=1),
    ),
}

df = (
    lk.dedupe(df)
    .apply(collection)
    .drop_duplicates(keep="first")
)
```

In the above: deduplicate records that share an exact email; and separately, deduplicate records whose addresses are fuzzy-similar at 0.98, then token-similar at 0.9. Records caught by either deduper end up in one cluster.

Keys run in insertion order, and the dedupers in a tuple value run in their listed order.


## Pipelines of Dedupers

Liken exposes a pipeline builder function for you to build complex, composable pipelines.

At a minimum, pipelines can replicate a dictionary collection. Compare the following two which produce identical outputs:

=== "Dict"

    ```python
    import liken as lk

    collection = {
        "email": lk.exact(),
        "address": (
            lk.fuzzy(threshold=0.98),
            lk.tfidf(threshold=0.9, ngram=(1, 2), topn=1),
        ),
    }

    df = (
        lk.dedupe(df)
        .apply(collection)
        .drop_duplicates(keep="first")
    )
    ```

=== "Pipeline"

    ```python
    import liken as lk

    pipeline = (
        lk.pipeline()
        .step(lk.col("email").exact())
        .step(lk.col("address").fuzzy(threshold=0.98))
        .step(lk.col("address").tfidf(threshold=0.9, ngram=(1, 2), topn=1))
    )

    df = (
        lk.dedupe(df)
        .apply(pipeline)
        .drop_duplicates()
    )
    ```

A pipeline has the following features:

- Each `step` in a pipeline represents a deduplication step.
- Column access is provided by `lk.col` expression.
- Dedupers are provided as method calls to the `lk.col` expression.

### How a pipeline executes

TODO move to common recipes

Each step produces clusters of linked records. Steps combine as OR: records linked by any step end up in one cluster. But clusters are not recomputed from scratch at the end — each step rewrites the canonical ids built so far, so the order of steps can change the outcome.

 id   |    x     |    y
------|----------|---------
  0   |    a     |    m
  1   |    a     |    n
  2   |    z     |    n

/// caption
Three records: `x` links rows 0 and 1; `y` links rows 1 and 2.
///

With `x` first, step `x` merges rows 0 and 1; step `y` then merges rows 1 and 2, and row 2 inherits row 1's canonical id — all three records share one id. With `y` first, row 2's cluster forms before `x` runs — and `x` never touches it, so row 2 keeps its own id. Run both orders and compare:

```python
import liken as lk

xy = lk.pipeline().step(lk.col("x").exact()).step(lk.col("y").exact())
yx = lk.pipeline().step(lk.col("y").exact()).step(lk.col("x").exact())

lk.dedupe(df).apply(xy).canonicalize().collect()  # canonical_id: [0, 0, 0]
lk.dedupe(df).apply(yx).canonicalize().collect()  # canonical_id: [0, 0, 1]
```

The same applies to dictionaries: keys run in insertion order. If your dedupers are truly independent — no record is linked by more than one deduper — order does not matter. When dedupers overlap, order them from the broadest to the narrowest.

### AND semantics

Pipelines support combining the effects of multiple dedupers using implicit and statements.

AND semantics are supported in Liken when lists of dedupers are passed to a `step` in a pipeline. For a step of threshold dedupers, a pair of records is linked only when every deduper in the step matches it. For a step mixing predicate and threshold dedupers, the step runs as a chain: the predicates match first, and each later deduper sees only the records matched so far — so records are linked only when every condition holds on the way through:

```python
import liken as lk

pipeline = (
    lk.pipeline()
    .step(
        [
            lk.col("address").fuzzy(threshold=0.8),
            lk.col("address").str_startswith("10"),
        ] # AND: both conditions must hold
    )
    .step(lk.col("email").fuzzy(threshold=0.98))
)

df = (
    lk.dedupe(df)
    .apply(pipeline)
    .drop_duplicates()
)
```

In the above case, for the first step, both conditions must hold — similar addresses will only be deduplicated if they also start with "10". See the difference on a small dataset:

 id   |    address
------|----------------
  0   |  10 high st
  1   |  10 high street
  2   |  99 high st
  3   |  99 high street

/// caption
Four addresses. The "10" pair and the "99" pair are both fuzzy-similar above 0.8.
///

With `fuzzy(0.8)` alone, all four records merge into one. With the AND step, the predicate rules out the "99" pair:

 id   |    address
------|----------------
  0   |  10 high st
  2   |  99 high st
  3   |  99 high street

/// caption
AND step output: only rows 0 and 1 were linked; the "99" records survive as two rows.
///

AND steps are most effective when one of the dedupers is a *predicate* deduper. Liken optimises such steps with *predicate evaluation*: the predicate dedupers run first, and the remaining dedupers only see the records the predicates matched. Liken reorders the step for you — the step is stably sorted so predicates run first — so you do not need to order dedupers yourself. A record counts as matched by a predicate only when the predicate actually links it to another record, and predicates see only the records the earlier predicates kept, so in a step with several predicates the order can matter. The optimisation matters because predicate dedupers run in about O(n) time while threshold dedupers compare every pair of records at about O(n²): restricting the threshold deduper to the predicate-matched subset skips the quadratic cost over records that could never match.

### OR semantics

OR semantics are captured by distinct steps in a pipeline.

OR semantics are actually implicitly supported when using dictionaries and are best understood in comparison with AND semantics:

=== "OR"
    ```python
    import liken as lk

    pipeline = (
        lk.pipeline()
        .step(lk.col("address").fuzzy())
        .step(lk.col("address").str_len(min_len=10))
    ) # OR: either condition must hold
    ```

=== "AND"
    ```python
    import liken as lk

    pipeline = (
        lk.pipeline()
        .step(
            [
                lk.col("address").fuzzy(),
                lk.col("address").str_len(min_len=10),
            ]
        )
    ) # AND: both conditions must hold
    ```

??? info "OR in dictionaries "
    OR semantics are achieved with dictionaries. If you are just using OR semantics in a pipeline, consider sticking to defining collections of dedupers as dictionaries, which are simpler to use.

### NOT semantics

Predicate dedupers can be inverted to form a NOT semantic by using the `~` operator on the column accessor `lk.col` expression:

```python
import liken as lk

pipeline = (
    lk.pipeline()
    .step(
        [
            lk.col("address").fuzzy(),
            ~lk.col("address").isna(), # NOT missing
        ]
    )
)
```

A negated predicate links the records that *do not* satisfy it. Watch it isolate missing values on a small dataset:

 id   |  address
------|-----------
  0   |  london
  1   |  london
  2   |  london
  3   |  None
  4   |  None
  5   |  tokyo

/// caption
Six addresses: three identical, two missing, one distinct.
///

The step `~lk.col("address").isna()` links every record whose address is *not* missing — rows 0, 1, 2 and 5 — so the three london records merge and tokyo stays apart:

 id   |  address
------|-----------
  0   |  london
  3   |  None
  4   |  None

/// caption
`~isna` output: the london records merged; the missing records were never in the cluster.
///

The un-negated `lk.col("address").isna()` is the mirror image: the two missing records merge into one, and every other record survives untouched.

### Preprocessors

Pipelines support the addition of a powerful feature: preprocessors. Liken's preprocessors transform data solely within the internals of the library for the purposes of deduplication whilst still returning data to you in the original format. Preprocessors never modify the data you get back — they run inside matching only, and your returned DataFrame keeps its original values.

Preprocessors can be used to refine deduplication pipelines, reduce boilerplate preprocessing code, reduce the number of "dummy" columns that you have to maintain, and reduces the risk of unacceptable false positive rates.

Preprocessors are available in the [`liken.preprocessors`](../reference/preprocessors.md) module and can be made available to the overall pipeline scope, a `step` in the pipeline, or `on` column only.

=== "Pipeline level"

    ```python
    import liken as lk

    pipeline = (
        lk.pipeline(preprocessors=lk.preprocessors.lower())
        .step(
            [
                lk.col("email").fuzzy(),
                ~lk.col("email").isna(),
            ],
        )
        .step(lk.col("address").tfidf())
    )

    ```

=== "Step level"

    ```python
    import liken as lk

    pipeline = (
        lk.pipeline()
        .step(
            [
                lk.col("email").fuzzy(),
                ~lk.col("email").isna(),
            ],
            preprocessors=lk.preprocessors.lower()
        )
        .step(lk.col("address").tfidf())
    )
    ```

=== "Col level"

    ```python
    import liken as lk

    pipeline = (
        lk.pipeline()
        .step(
            [
                lk.col("email").fuzzy(),
                ~lk.col("email").isna(),
            ],
        )
        .step(lk.col("address", preprocessors=lk.preprocessors.lower()).tfidf())
    )
    ```

A single preprocessor can be passed, or multiple, if passed as a list:

```python
import liken as lk

pipeline = (
    lk.pipeline(
        preprocessors=[
            lk.preprocessors.lower(),
            lk.preprocessors.ascii_fold(),
            lk.preprocessors.remove_punctuation(),
        ]
    )
    .step(lk.col("address").tfidf())
)
```

Preprocessors are propagated in a top-down manner, but overridden bottom-up. So, a `pipeline` level preprocessor will propagate to each `step` and column accessor `on`, but will be respectively overridden if preprocessors are defined there:

```python
pipeline = (
    lk.pipeline(preprocessors=[lk.preprocessors.ascii_fold()])
    .step(
        [
            lk.col("email").fuzzy(),  # preprocessed by step's preprocessor, `alnum`.
            ~lk.col(
                "address",
                preprocessors=[lk.preprocessors.lower()],
            ).isna(),  # uses its own preprocessor, `lower`.
        ],
        preprocessors=[lk.preprocessors.alnum()],  # defines the step's preprocessor
    )
    .step(
        lk.col("address").tfidf()
    )  # defaults to the pipeline's preprocessor, `ascii_fold`.
)
```
