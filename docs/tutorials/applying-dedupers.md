---
title: Applying Dedupers
---

In the [First Steps](../tutorials/first-steps.md#the-simplest-example) you found out how to replicate exact deduplication with **liken** — in fact it was *the* [exact](../reference/liken.md#liken.exact) deduper in use. It came bundled with `dedupe` when you called `.drop_duplicates` with no other deduper.

To use a [built-in deduper](./first-steps.md#built-in-dedupers), the deduper is *applied* with `.apply`.

```python
import liken as lk

df = (
    lk.dedupe(df)
    .apply(lk.fuzzy())
    .drop_duplicates("address")
    .collect()
)
```

## Single Dedupers

If you only need a single deduper, use it straight in `.apply` as seen above. The column or columns to dedupe on are passed in `.drop_duplicates`.

## Dictionaries of Dedupers

**Liken** supports deduplicating with a collection of dedupers. This allows deduplicating multiple columns with different dedupers, or defining several dedupers to be run sequentially on a column.

The **Liken** solves this with dictionaries. `.drop_duplicates` no longer accepts a column label argument — columns will now be defined as the keys to the dictionary. 

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
    .collect()
)
```

In the above case, the defined collection reads as "Deduplicate exact emails. Then, similar addresses using Fuzzy and then TF-iDF. Finally, any deduplicate records that have 3 out of 4 of those categories matching".

!!! note "`keep` arg"
    The `keep` argument accepts the literals "first" or "last" which defines which record will be kept from a duplicate set of records, based on their position in the dataframe.

## Pipelines of Dedupers

**Liken** exposes a pipeline builder function for you to build complex, composable pipelines. 

At a minimum, pipelines can replicate a dictionary collection. For example, the dictionary collection we saw above can be instead represented as:

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
    .collect()
)
```

A pipeline has the following features:

- Each `step` in a pipeline represents a deduplication step.
- Column access is provided by `lk.col` expression.
- Dedupers are provided as method calls to the `lk.col` expression.

### AND semantics

Pipelines support combining the effects of multiple dedupers using implicit and statements.

AND semantics are supported in **Liken** when lists of dedupers are passed to a `step` in a pipeline. All conditions in a step must match for records to be linked by the pipeline:

```python
import liken as lk

pipeline = (
    lk.pipeline()
    .step(
        [
            lk.col("address").fuzzy(),
            lk.col("address").str_len(min=10),
        ] # AND: both conditions must hold
    )
    .step(lk.col("email").fuzzy(threshold=0.98))
)

df = (
    lk.dedupe(df)
    .apply(pipeline)
    .drop_duplicates()
    .collect()
)
```

In the above case, for the first step, both conditions must hold — similar addresses will only be deduplicated if the length of addresses has a minimum length of 10 characters.

??? info "Effective combinations of dedupers"
    AND semantics are supported between any **Liken** deduper but are especially effective when combining a similarity deduper with a predicate deduper. Additionally, **Liken** features an optimisation, "Rule Predication" which enforces the execution of the predicate deduper first given an AND semantic `step` — the subsequent dedupers will then only operate on the subset of identical records collected by the first predicate deduper. This optimisation works becase by their nature predicate dedupers operate close to *O(n)* whilst similarity dedupers generally operate at *O(^n^2)*.

### OR semantics

OR semantics behaviour is captured by distinct steps in a pipeline. 

OR semantics are actually implicitely supported when using dictionaries and are best understood in comparison with AND semantics:

=== "OR"
    ```python
    import liken as lk

    pipeline = (
        lk.pipeline()
        .step(lk.col("address").fuzzy())
        .step(lk.col("address").str_len(min=10))
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
                lk.col("address").str_len(min=10),
            ]
        )
    ) # AND: both conditions must hold
    ```

??? info "OR in dictionaries "
    OR semantics are achieved in with dictionaries. If you are just using OR semantics in pipeline, consider sticking to defining collections of dedupers are dictionaries, which are simpler to use.

### NOT semantics

Predicate dedupers can be inverted to form a NOT semantic by using the `~` operator on the column accessor `lk.col` expression:

```python
import liken as lk

pipeline = (
    lk.pipeline()
    .step(
        [
            lk.col("address").fuzzy(),
            ~lk.col("address").isna(), # NOT null
        ]
    )
)
```

### Preprocessors

Pipelines support the addition of a powerful feature: preprocessors. **Liken's** preprocessors transform data solely within the internals of the library for the purposes of deduplication whilst still returning data to you in the original format. 

Preprocessors can be used to refine deduplication pipelines, reduce boilerplate code preprocessing code, reduce the number of "dummy" columns that you have to maintain, and reduces the risk of unacceptable false positive rates.

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

Preprocessors are propagated in a top-down manner, but overriden buttom-up. So, a `pipeline` level preprocessor will propagate to each `step` and column accessor `on`, but will be respectively overriden if preprocessors are defined there:

```python
pipeline = (
    lk.pipeline(preprocessors=[lk.preprocessors.ascii_fold()])
    .step(
        [
            lk.col("email").fuzzy(),  # preprocessed by step's preprocessor, `alnum`.
            ~lk.col(
                "address",
                preprocessors=[lk.preprocessors.lower()],
            ).isna(),  # uses it's own preprocessor, `lower`.
        ],
        preprocessors=[lk.preprocessors.alnum()],  # defines the step's preprocessor
    )
    .step(
        lk.col("address").tfidf()
    )  # defaults to the pipeline's preprocessor, `ascii_fold`.
)
```

## Summary

Different collections of dedupers, whether a single deduper, a dictionary or a pipeline, are best suited to different use cases:

| Collection | Quick tasks | Multiple columns | OR semantics | AND semantics | NOT semantics | Preprocessors |
| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Single | :white_check_mark: | :material-close: | :material-close: | :material-close: | :material-close: | :material-close: |
| Dict | :white_check_mark: | :white_check_mark: | :white_check_mark: | :material-close: | :material-close: | :material-close: |
| Pipeline | :material-close: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
