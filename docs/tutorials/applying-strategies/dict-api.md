---
title: Dict API
---

## Dictionary based Strategies

Earlier we explored the Sequential API. Once you've applied more than just a couple of strategies you're pushing verbosity to the limit — once you need to so for *multiple* columns it's getting really quite impractical.

The **Dict API** solves this with dictionaries. Here you'll only be able to use `apply` once. Additionally, `drop_duplicates` will no longer accept any arguments — columns will now be defined as the keys to the dictionary. Let's deduplicate our persons dataset on more than just `address`:

```python
from enlace import Dedupe, exact, fuzzy, tfidf, jaccard

STRAT = {
    "email": exact(),
    "address": (
        fuzzy(threshold=0.98),
        tfidf(threshold=0.9, ngram=(1, 2), topn=1),
    ),
    (
        "marital_status",
        "has_car",
        "flat_or_house",
    ): jaccard(threshold=0.8),
}

dp = Dedupe(df)
dp.apply(STRAT)     
df = dp.drop_duplicates(keep="first")
```

Wow! In plain English this now might read as "Deduplicate only exact emails. Then similar addresses. Finally, any records that have 3 out of 4 of those categories matching".

!!! note "`keep` arg"
    The `keep` argument accepts the literals "first" or "last" which defines which record will be kept.

!!! tip
    In the example above we defined the strategy as a constant so it's clear that not much is going on with the `Dedupe` class. Hopefully this helps to highlight that it's on *you* to define the best strategy possible given your dataset — and that with the **Dict API**, that is remarkably easy.


## Recap

If you're only ever looking to aim to *drop* duplicates then you can skip the next section on [**Record Linkage**](../../concepts/record-linkage.md).

If the above is as complex a strategy as you'll ever use, then stop here.

!!! success "You learnt:"
    - Multiple strategies for multiple columns can be defined using a dictionaty, thus forming the **Dict API**.
    - The **Dict API** only allows a single call to `.apply`