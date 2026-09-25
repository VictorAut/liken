---
title: Applying Dedupers
---

In the [First Steps](../tutorials/first-steps.md#the-simplest-example) you found out how to replicate exact deduplication with **Liken** — in fact it was *the* [exact](../reference/liken.md#liken.exact) deduper in use. It came bundled with `dedupe` when you called the `drop_duplicates` function with no other deduper.

To use a [built-in deduper](./first-steps.md#built-in-dedupers), a deduper is *applied* with the `apply` function:

```python
import liken as lk

df = (
    lk.dedupe(df)
    .apply(lk.fuzzy())
    .drop_duplicates("address")
)
```

## Single Dedupers

If you only need a single deduper, use it straight in an `apply` function as seen above, i.e. `apply(lk.fuzzy())`, or any other deduper. The column or columns to dedupe on are passed in `drop_duplicates`.

Usage of single dedupers is limited — you can only ever use a single deduper on a single set of columns.

### Coming from Pandas?

When it comes to single dedupers, as above, **Liken** is easy to use, but especially so if you are coming from Pandas. Special affordances have been made to supply you with the means to use Pandas's `drop_duplicates` in a "fuzzy manner". To do this, simply import `liken`, and pass the deduper as an accessor to your pandas dataframe. Any keyword arguments that usually get passed to the deduper, now simply get passed to `drop_duplicates`:

=== "With Pandas Affordance"

    ```python
    import liken as lk # Only works if you import liken!
    import pandas as pd

    df = pd.read_csv("...")

    df = df.fuzzy.drop_duplicates("address", threshold=0.6) # (1)!
    ```

    1.  `kwargs` for `fuzzy` are delegated to `drop_duplicates`. This is also true for other dedupers; for example, when using the `tfidf` deduper, the `ngram` kwarg would be passed in `drop_duplicates` too i.e. `df.tfidf.drop_duplicates("address", threshold=0.6, ngram=3)`

=== "Normal API Use"

    ```python
    import liken as lk
    import pandas as pd

    df = pd.read_csv("...")

    df = (
        lk.dedupe(df)
        .apply(lk.fuzzy(threshold=0.6)) # (1)!
        .drop_duplicates("address")
    )
    ```

    1.  `kwargs` are applied in the function, as defined by the [API](../reference/liken.md)


Pandas affordances are limited to [fuzzy](../reference/liken.md#liken.fuzzy), [tfidf](../reference/liken.md#liken.tfidf), [lsh](../reference/liken.md#liken.lsh), [jaccard](../reference/liken.md#liken.jaccard), and [cosine](../reference/liken.md#liken.cosine). Also, this special use is limited to single dedupers, and does not support the application of collections of dedupers, as shown next.

??? info "Pandas affordances"
    **Liken's** Pandas extension is only useable if you actually *import* `liken`!

## Collections of Dedupers

**Liken** supports deduplicating with a collection of dedupers. This allows:

- Deduplicating multiple sets of columns with different dedupers
- Defining several dedupers to be run sequentially on a set of columns

Collections are supported in two formats. Dictionaries provide quick and easy composability; pipelines provide fully-featured composability with support for logical rules and built-in preprocessors.

### Dictionaries

When defining a collection as a dictionary, `drop_duplicates` no longer accepts a column label argument — columns will now be defined as the keys to the dictionary.

Dictionaries have three rules:

- **Keys are column sets.** A key is a column label (`"email"`) or a tuple of labels (`("first_name", "last_name")`) — the columns its rule runs on.
- **Tuple values are sequential steps.** A value is one deduper, or a tuple of dedupers chained on that key's columns: a record counts as a duplicate if it is linked by any deduper in the tuple.
- **Keys combine as OR.** Records linked by *any* key's rule end up in the same group.

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

The collection above reads as: deduplicate records that share an exact email; and separately, deduplicate records whose addresses are fuzzy-similar at 0.98, then token-similar at 0.9. Records caught by either rule end up in one group.

Keys run in insertion order, and the dedupers in a tuple value run in their listed order — both can matter; see [How a pipeline executes](#how-a-pipeline-executes).

??? info "`keep` arg"
    The `keep` argument accepts the literals "first" or "last" which defines which record will be kept from a duplicate set of records, based on their position in the dataframe.

### Pipelines of Dedupers

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
)
```

A pipeline has the following features:

- Each `step` in a pipeline represents a deduplication step.
- Column access is provided by `lk.col` expression.
- Dedupers are provided as method calls to the `lk.col` expression.

### How a pipeline executes

Each step produces groups of linked records. Steps combine as OR: records linked by any step end up in one group. But groups are not recomputed from scratch at the end — each step rewrites the canonical ids built so far, so the order of steps can change the outcome.

 id   |    x     |    y
------|----------|---------
  0   |    a     |    m
  1   |    a     |    n
  2   |    z     |    n

/// caption
Three records: `x` links rows 0 and 1; `y` links rows 1 and 2.
///

With `x` first, step `x` merges rows 0 and 1; step `y` then merges rows 1 and 2, and row 2 inherits row 1's canonical id — all three records share one id. With `y` first, row 2's group forms before `x` runs — and `x` never touches it, so row 2 keeps its own id. Run both orders and compare:

```python
import liken as lk

xy = lk.pipeline().step(lk.col("x").exact()).step(lk.col("y").exact())
yx = lk.pipeline().step(lk.col("y").exact()).step(lk.col("x").exact())

lk.dedupe(df).apply(xy).canonicalize().collect()  # canonical_id: [0, 0, 0]
lk.dedupe(df).apply(yx).canonicalize().collect()  # canonical_id: [0, 0, 1]
```

/// caption
Step order changes which records survive `drop_duplicates`.
///

The same rule applies to dictionaries: keys run in insertion order. If your rules are truly independent — no record is linked by more than one rule — order does not matter. When rules overlap, order them from the broadest to the narrowest.

#### AND semantics

Pipelines support combining the effects of multiple dedupers using implicit and statements.

AND semantics are supported in **Liken** when lists of dedupers are passed to a `step` in a pipeline. For a step of similarity dedupers, a pair of records is linked only when every deduper in the step matches it. For a step mixing predicate and similarity dedupers, the step runs as a chain: the predicates match first, and each later deduper sees only the records matched so far — so records are linked only when every condition holds on the way through:

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

With `fuzzy(0.8)` alone, all four records collapse to one row. With the AND step, the predicate rules out the "99" pair:

 id   |    address
------|----------------
  0   |  10 high st
  2   |  99 high st
  3   |  99 high street

/// caption
AND step output: only rows 0 and 1 were linked; the "99" records survive as two rows.
///

AND steps are most effective when one of the dedupers is a *predicate* deduper. **Liken** optimises such steps with *rule predication*: the predicate dedupers run first, and the remaining dedupers only see the records the predicates matched. **Liken** reorders the step for you — the step is stably sorted so predicates run first — so you do not need to order dedupers yourself, and results are identical either way. The optimisation matters because predicate dedupers run in about O(n) time while similarity dedupers compare every pair of records at about O(n²): restricting the similarity deduper to the predicate-matched subset skips the quadratic cost over records that could never match.

#### OR semantics

OR semantics behaviour is captured by distinct steps in a pipeline.

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

#### NOT semantics

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

A negated predicate groups the records that *do not* satisfy it. Watch it isolate nulls on a small dataset:

 id   |  address
------|-----------
  0   |  london
  1   |  london
  2   |  london
  3   |  None
  4   |  None
  5   |  tokyo

/// caption
Six addresses: three identical, two null, one distinct.
///

The step `~lk.col("address").isna()` groups every record whose address is *not* null — rows 0, 1, 2 and 5 — so the three london records collapse and tokyo stays apart:

 id   |  address
------|-----------
  0   |  london
  3   |  None
  4   |  None

/// caption
`~isna` output: the london records merged; the null records were never in the group.
///

The un-negated `lk.col("address").isna()` is the mirror image: the two null records collapse to one, and every other record survives untouched.

#### Preprocessors

Pipelines support the addition of a powerful feature: preprocessors. **Liken's** preprocessors transform data solely within the internals of the library for the purposes of deduplication whilst still returning data to you in the original format. Preprocessors never modify the data you get back — they run inside matching only, and your returned DataFrame keeps its original values.

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

## Growing a Pipeline

The features above compose into a way of working: start narrow, observe what the rules miss, widen carefully, and qualify what overshoots. This section walks through that on one small dataset:

 id   |     name      |        email
------|---------------|----------------------
  1   |  Alice Wong   |     a.wong@ex.io
  2   |  alice wong   |     a.wong@ex.io
  3   |  Alicia Wong  |  alicia.wong@ex.io
  4   | Bob Cratchit  |  bob.cratchit@ex.io
  5   | Bob Cratchet  |  bob.cratchet@ex.io

/// caption
Five customers: a case variant (rows 1–2), a false friend (row 3), a spelling variant (rows 4–5).
///

**Start with exact matching.** It merges nothing — all five names are distinct strings, and `alice wong` misses `Alice Wong` on case alone:

```python
import liken as lk

df = lk.dedupe(df).drop_duplicates("name")
```

**Widen to fuzzy.** At threshold 0.7, `fuzzy` catches the case variant (0.80) and both real duplicate pairs — and one record too many: `Alicia Wong` scores 0.86 against `Alice Wong`, so a different person is merged:

```python
import liken as lk

pipeline = lk.pipeline().step(lk.col("name").fuzzy(threshold=0.7))

df = lk.dedupe(df).apply(pipeline).drop_duplicates()
```

 id   |     name      |        email
------|---------------|----------------------
  1   |  Alice Wong   |     a.wong@ex.io
  4   | Bob Cratchit  |  bob.cratchit@ex.io

/// caption
After the fuzzy step: 5 records down to 2. Alicia Wong was absorbed by mistake.
///

**Qualify the match.** The false positive shares the surname but not the address. Requiring the emails to agree too splits her back out:

```python
import liken as lk

pipeline = lk.pipeline().step(
    [
        lk.col("name").fuzzy(threshold=0.7),
        lk.col("email").fuzzy(threshold=0.9),
    ] # AND: both conditions must hold
)

df = lk.dedupe(df).apply(pipeline).drop_duplicates()
```

 id   |     name      |        email
------|---------------|----------------------
  1   |  Alice Wong   |     a.wong@ex.io
  3   |  Alicia Wong  |  alicia.wong@ex.io
  4   | Bob Cratchit  |  bob.cratchit@ex.io

/// caption
After the AND step: 3 records. `Alicia Wong` survives as her own record; both real duplicate pairs merged.
///

That is the loop: exact to see the baseline, fuzzy to catch the near-misses, and a second condition to protect against over-merging. `explore` gives you the duplicate-rate evidence to pick each threshold; the next tutorial applies the same loop to record linkage.

## Summary

Different collections of dedupers, whether a single deduper, a dictionary or a pipeline, are best suited to different use cases:

| Collection | Pandas extension | Quick tasks | Multiple columns | Logical rule semantics | Preprocessors |
| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Single | :white_check_mark: | :white_check_mark: | :material-close: | :material-close: | :material-close: |
| Dict | :material-close: | :white_check_mark: | :white_check_mark: | :material-close: | :material-close: |
| Pipeline | :material-close: | :material-close: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
