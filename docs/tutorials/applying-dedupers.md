---
title: Applying Dedupers
---

## Applying Your First Deduper

In your [First Steps](../tutorials/first-steps.md#the-simplest-example) you found out how to replicate exact deduplication with **liken** — in fact it was *the* [exact](../reference/liken.md#liken.exact) deduper in use. It came bundled with `dedupe` when you called `drop_duplicates()` with no other deduper.

When it comes to 'using' dedupers, they are *applied* first, before calling `drop_duplicates()`:


```python
import liken as lk

df = lk.dedupe(df).apply(fuzzy()).drop_duplicates("address").collect()
```

Throughout **Liken** when you want to use a deduper, you always use `apply`. This is true regardless of the type of deduper (i.e. similarity based or predicate based).

## Dictionaries of Dedupers

We just saw how to deduplicate a dataframe having applied a *single* deduper. What if you want to deduplicate with a collection of dedupers?

The **Liken** solves this with dictionaries. `drop_duplicates()` no longer accepts a column label argument — columns will now be defined as the keys to the dictionary. 

```python
import liken as lk

collection = {
    "email": lk.exact(),
    "address": (
        lk.fuzzy(threshold=0.98),
        lk.tfidf(threshold=0.9, ngram=(1, 2), topn=1),
    ),
    (
        "marital_status",
        "has_car",
        "flat_or_house",
        "urban_or_rural",
    ): lk.jaccard(threshold=0.75),
}

df = lk.dedupe(df).apply(collection).drop_duplicates(keep="first").collect()
```

When using dictionaries, each key defines a deduper for a column of data dataframe. Dedupers can be passed as a tuple in which case each deduper will be used sequentially.
In the above case, the defined collection reads as "Deduplicate exact emails. Then, similar addresses using Fuzzy and then TF-iDF. Finally, any deduplicate records that have 3 out of 4 of those categories matching".

!!! note "`keep` arg"
    The `keep` argument accepts the literals "first" or "last" which defines which record will be kept from a duplicate set of records, based on their position in the dataframe.

## Pipelines of Dedupers

**Liken** exposes a pipeline builder function for you to build complex, composable pipelines. Above we saw that we dedupers can be defined as collections using a dictionary construct. Here, we first replicate that functionality but using the `liken.pipeline` module instead:

```python
import liken as lk

pipeline = (
    lk.pipeline()
    .step(lk.col("email").exact())
    .step(lk.col("address").fuzzy(threshold=0.98))
    .step(lk.col("address").tfidf(threshold=0.9, ngram=(1, 2), topn=1))
    .step(
        lk.col(
            (
                "marital_status",
                "has_car",
                "flat_or_house",
                "urban_or_rural",
            ).jaccard(threshold=0.8)
        ),
    )
)

df = lk.dedupe(df).apply(pipeline).drop_duplicates().collect()
```

A pipeline has the following features:

- Each `step` in a pipeline represents a deduplication step
- Column access is provided by `lk.col`.
- Dedupers are provided as method calls to `lk.col`.

### AND semantics

Consider an "address" column. Deduplicating with a similarity based dedupers, such as `fuzzy`, might get some good results. However, a typical "address" column might have several conditions that make a uniform application of a similarity deduper inadequate. For instance, the "address" column might contain nulls, which would typically be considered idnetical records. Alternatively, addresses might vary greatly in length, to the point that short addresses might be deduplicated too aggresively by a low `threshold` parameter in a deduper, thus yielding too many false positives. In such case you would be motivated to only apply a similarity deduper in conjunction with another deduper such that deduplication only happens when the formaer AND the later agree. 

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


df = lk.dedupe(df).apply(pipeline).canonicalize().collect()
```

In the above case, for the first step, both conditions must hold — similar addresses will only be deduplicated if the length of addresses has a minimum length of 10 characters.

Although it might appear that AND semantics are akin to filters, this is not true in **Liken's** internals. AND semantics are only supported *between dedupers*. Filter's are not supported because no provision is ever made to drop data. In that regard, what might appear as a filter, for example the `isna()` function, is a actually a *predicate* deduper whose action is to canonicalize all records that fit the null description.

!!! info
    AND semantics are supported between any **Liken** deduper but are especially effective when combining a similarity deduper with a predicate deduper. Additionally, **Liken** features an optimisation, "Rule Predication" which enforces the execution of the predicate deduper first given an AND semantic `step` — the subsequent dedupers will then only operate on the subset of identical records collected by the first predicate deduper. This optimisation works becase by their nature predicate dedupers operate close to *O(n)* whilst similarity dedupers generally operate at *O(^n^2)*.

### OR semantics

OR semantics behaviour is captured by distinct steps in a pipeline. 

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
```python
import liken as lk

pipeline = (
    lk.pipeline()
    .step(lk.col("address").fuzzy())
    .step(lk.col("address").str_len(min=10))
) # OR: either condition must hold
```
!!! note
    OR semantics are achieved in with dictionaries. If you are just using OR semantics in pipeline, consider sticking to defining collections of dedupers are dictionaries, which are simpler to use.

### NOT semantics

Predicate dedupers can be inverted to form a NOT semantics by using the `~` operator on the column accessor `lk.col`. If in our earlier example we wanted to deduplicate where the minimum length was *not* 20 characters then we would have the following:

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
