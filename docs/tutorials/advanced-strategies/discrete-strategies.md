---
title: Discrete Strategies
---

## More About "and" Statements

Earlier, in [Boolean Operators](../advanced-strategies/boolean-operators.md) we used the idea that we could combine (`&`) strategies. What if instead of combining the strategies that by now you may know and love so much we instead combined an altogether different class of strategy?

## Discrete Outputs

Boolean algebra by combining strategies using `&` statements would work best if we could *modify* the behaviour of a strategy against a binary (discrete output). The strategies that we have so far used are "at threshold" deduplication strategies, namely, strategies that will deduplicate upon exceeding a minimum similarity threshold.

We can imagine an altogether different class of strategy that actually deduplicates based on a binary (discrete) decisions. For example, imagine that we deduplicate an `address` column based on whether or not the length of the address string exceeded 20 characters. Not super useful by itself — it's hard to see the value in doing that — but a powerful approach when combined with our "traditional" near deduplication strategies. This binary deduper, [`str_len`](../../../reference/rules/#enlace.rules.str_len) is actually available as part of the `enalce.rules` sub-package and more widely the **Rules API**:

```python {hl_lines="2 5"}
from enlace import Dedupe, fuzzy
from enlace.rules import Rules, on, str_len

STRAT = Rules(
    on("address", fuzzy(threshold=0.8)) 
    & on("address", str_len(min_len=20))
)

dp = Dedupe(df)
dp.apply(STRAT)     
df = dp.drop_duplicates()
```

Great! That strategy now let's you achieve near deduplication for addresses fields that *also* have the requirement of having a minimum length!

!!! tip
    Combining similarity ("at threshold") strategies with binary choice strategies is an especially useful combination. Consider above a dataset that may have had a feature engineered field for `address` that contained some very short addresses. The likelyhood of false positives when tuning a strategy to work well for "real" addresses is all to real, given what may be a plethora of short addresses. But the *modification* of that approach by using a binary choice strategy makes your deduplication robust.


## Inversions

Any discrete output should be able to be inverted to form a "not" statement. With discrete strategies from the `enlace.rules` package it is always possible to invert the functionality of the strategy using the `~` operator directly on the strategy. If in our earlier example we wanted to deduplicate where the minimum length was *not* 20 characters then we would have the following:

```python {hl_lines="3"}
STRAT = Rules(
    on("address", fuzzy(threshold=0.8)) 
    & on("address", ~str_len(min_len=20))
)
```

## Putting it All Together

Let's build a comprehensive strategy that might address the deduplication/canonicalization needs of a persons dataset. Last time we did this was in the [Dict API tutorial](../applying-strategies/dict-api.md). We'll aim to do something similar, but adding rules:

```python
from enlace import Dedupe, exact, fuzzy, jaccard
from enlace.rules import Rules, on, isna, str_len, str_contains

STRAT = Rules(
    on("email", exact()) & on("email", ~isna()),
    on("address", fuzzy(threshold=0.8))
    & on("email", isna())
    & on(
        "address",
        str_len(min_len=15),
    ),
    on(
        (
            "marital_status",
            "has_car",
            "flat_or_house",
        ),
        jaccard(threshold=0.8),
    )
    & on("postcode", str_contains(r"90\d{3}", regex=True)),
)

dp = Dedupe(df)
dp.apply(STRAT)     
df = dp.drop_duplicates()
```

Great! Well done! You can now build highly customizable deduplication (or canonicalization) pipelines!

!!! tip
    Again, note that `&` combinations can chain any number of `on` executors. Also, note that the respective `on`s do not need to be acting on the same column. In the above example, the `address` column was only deduplicated when the `email` was not null, for example.

## Performance

Deduplicating a dataset can be a compute-intensive task so there might be a balance to strike when adding strategies, as each added strategy is additional compute time. The next tutorial on [Supported Backends](../supported-backends.md) might offer a path forwards for very large datasets or other optimisation needs you might have.

!!! warning
    It's worth noting that discrete strategies are a bit like a `WHERE` statement. However, they do not currently work like that — if they did then you would have the guarantee that a subsequent strategy, especially a similarity at threshold one would operate on a subset of the dataset. This is due for a future implementation. For the time being **Enlace** will have to handle the workload of a deduplicating the dataset twice (or more).

## All of **Enlace's** Strategies

There are several binary (discrete) strategies available. They're listed here, appended to the list of strategies you saw beforehand in the [Deduplication Strategies tutorial](../strategies.md)

| |               | Strategy                                              | Description                                                                 |
|-------------| ------------- | ----------------------------------------------------- | ---------------------------------------------------------------- |
| *Continuous* |*single-column*| [`exact`](../../../reference/enlace/#enlace.exact)       | You've already seen this in use *implicitely* in your [First Steps](../first-steps.md)  |
| *Continuous* |*single-column*| [`fuzzy`](../../../reference/enlace/#enlace.fuzzy)       | Fuzzy string matching                                                                            |
| *Continuous* |*single-column*| [`tfidf`](../../../reference/enlace/#enlace.tfidf)       | String matching with Tf-Idf                                                                      |
| *Continuous* |*single-column*| [`lsh`](../../../reference/enlace/#enlace.lsh)           | String matching with Locality Sensitive Hashing (LSH)                                            |
| *Continuous* |*compound-column*| [`jaccard`](../../../reference/enlace/#enlace.jaccard) | Multi column similarity based on intersection of categorical data                                |
| *Continuous* |*compound-column*| [`cosine`](../../../reference/enlace/#enlace.cosine)   | Multi column similarity based on dot product of numerical data                                   |
| *Discrete* |*single-column*| [`isna`](../../../reference/rules/#enlace.rules.isna)                | Records where the column value is null/`None`                                        |
| *Discrete* |*single-column*| [`isin`](../../../reference/rules/#enlace.rules.isin)                | Records where the column value is in a list of members                               |
| *Discrete* |*single-column*| [`str_startswith`](../../../reference/rules/#enlace.rules.str_startswith)     | Records where the string starts with a pattern                                       |
| *Discrete* |*single-column*| [`str_endswith`](../../../reference/rules/#enlace.rules.str_endswith)       | Records where the string ends with a pattern                                         |
| *Discrete* |*single-column*| [`str_contains`](../../../reference/rules/#enlace.rules.str_contains)         | Records where the string contains a pattern. Accepts Regex.                          |
| *Discrete* |*single-column*| [`str_len`](../../../reference/rules/#enlace.rules.str_len)              | Records where the string length is bounded by a minimum and maximum length           |

!!! tip
    **Any** combination of strategies is possible! Whilst the use cases tend to favour combinating a continuous (similarity at threshold) strategy with a discrete strategy, you can also combine two discrete strategies if you want.

    In fact you can *chain any number of strategies with `on` executors!*


## Recap

!!! success "You learnt:"
    - That the `enlace.rules` sub-package provides it's own discrete strategies.
    - Discrete strategies can be powerfully used in combinations with similarity-at-threshold strategies.
    - Discrete strategy outcomes can be inverted with `~`.
    - `on` statements accept any number of combinations, for any combination of strategies, or columns.