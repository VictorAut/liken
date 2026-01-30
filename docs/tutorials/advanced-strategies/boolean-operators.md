---
title: Boolean Operators
---

## Strategies That Are Combined (`&`)

Let's again think about an `address` column. Deduplicating with the `fuzzy` strategy might get some good results and that might equally be true with the `lsh` strategy. But doing that may yield too many false positives and perhaps you find that the results are more accurate when both the `fuzzy` strategy *and* the `lsh` strategy agree.

Combining strategies with "and" statements (using the `&` operator) forms the basis of the motivation for the `liken.rules` sub-package. And statements are used to combine `on` executors. Above we described a problem statement — let's now translate that into Python code:

```python {hl_lines="5 6"}
from liken import Dedupe, fuzzy, lsh
from liken.rules import Rules, on

STRAT = Rules(
    on("address", fuzzy(threshold=0.8)) 
    & on("address", lsh(threshold=0.8, num_perm=256))
)

lk = Dedupe(df)
lk.apply(STRAT)     
df = lk.drop_duplicates()
```

Your telling **Liken** that it can only consider a record to be valid for deduplication if *both* the strategies agree.

## Or Strategy?

There's no such thing as an `|` strategy. Take a moment to realise that `|` is captured by comma separation in the `Rules API` in the same way that it is in the **Dict API**. No explicit functionality is provisioned for "or" statements. Note that the following two strategies are **not equivalent** and will produce **different results**:

```python
STRAT = Rules(
    on("address", fuzzy()),        # comma separated *is like* or
    on("address", lsh())
)
```
```python
STRAT = Rules(
    on("address", fuzzy())         # and combined
    & on("address", lsh())
)
```

## Recap

This tutorial doesn't *quite* logically end here. The [Discrete Strategies concepts tutorial](../advanced-strategies/discrete-strategies.md) will close the loop on the power of combinations.

!!! success "You learnt:"
    - `on` executors can be combined as and statements using the `&` operator.
    - There is no explicit provisioning of an or statement — comma separation of strategies inside `Rules` captures the logic.