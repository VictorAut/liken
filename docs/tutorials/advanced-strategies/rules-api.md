---
title: Rules API
---

??? question "Yet Another API...?"
    Yes. The **Rules API** defines advanced usage, and offers tooling that is unavailable to the **Sequential** and **Dict API**s. 

    This tutotial, and the next two define the full scope of the **Rules API**:

    - [Boolean Operators](../advanced-strategies/boolean-operators.md)
    - [Discrete Strategies](../advanced-strategies/discrete-strategies.md)


## Strategies as Rules

Let's first use the **Rules API** to replicate the complex strategy we saw as an example earlier in the tutorial on applying strategies as dictionaries in the [**Dict API** tutorial](../applying-strategies/dict-api.md). To do so we will introduce the `rules` sub-package:

```python {hl_lines="2"}
from enlace import Dedupe, exact, fuzzy, tfidf, jaccard
from enlace.rules import Rules, on

STRAT = Rules(
    on("email", exact()),
    on("address", fuzzy(threshold=0.98)),
    on("address", tfidf(threshold=0.9, ngram=(1, 2), topn=1)),
    on(
        (
            "marital_status",
            "has_car",
            "flat_or_house",
        ),
        jaccard(threshold=0.8),
    ),
)

dp = Dedupe(df)
dp.apply(STRAT)     
df = dp.drop_duplicates()
```

As you can see, organised like this, it's not too disimilar to the **Dict API**:

- Each `on` enacts a strategy *on* a column, a bit like key-value pairs in dictionary.
- Strategies are used in the order in which they are defined.
- The entire collection is wrapped in a `Rules` object.

We've not done anything new yet with this ability to define strategies, other than to get the minimum requirement that it feels and acts the same as a complex **Dict API** strategy. You may even prefer the look of this.

!!! note
    The **Rules API**, just like **Dict API**, also enforces the use of a single call to `apply()`. The emphasis is still on you to construct stragegies that appropriate model the deduplication (or canonicalization) needs of your DataFrame. In these tutorials we repeatedly build a `STRAT` constant to emphasise that fact.


## Collections of `on`

`Rules` don't accept length-2 tuples. They *need* `on`. You'll see why in the next tutorial.

## Recap

!!! success "You learnt:"
    - Multiple strategies for multiple columns can be defined using a single `Rules` with several `on` functions, forming the **Rules API**.
    - The **Rules API** only allows a single call to `apply()`