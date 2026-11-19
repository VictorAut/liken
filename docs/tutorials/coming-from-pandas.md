---
title: Coming from Pandas?
---

## Introduction

Liken was originally built to provide easy-to-use near deduplication as a single-line bolt-on to the Pandas API.

!!! warning

    Jump to [first steps](../tutorials/first-steps.md) if you need Liken's full suite of tooling to deduplicate Pandas DataFrames.

## Near Deduplication

Simply import `liken`, and pass the deduper as an accessor to your pandas dataframe. Any keyword arguments that usually get passed to the deduper, now simply get passed to `drop_duplicates`:

=== "Pandas API"

    ```python
    import pandas as pd

    df = pd.read_csv("...")

    df = df.drop_duplicates("address")
    ```

=== "With Liken with Pandas"

    ```python
    import liken as lk # (1)!
    import pandas as pd

    df = pd.read_csv("...")

    df = df.fuzzy.drop_duplicates("address", threshold=0.6) # (2)!
    ```

    1. Liken's Pandas extension is only useable if you actually *import* `liken`!
    2. `kwargs` for `fuzzy` are delegated to `drop_duplicates`. This is also true for other dedupers; for example, when using the `tfidf` deduper, the `ngram` kwarg would be passed in `drop_duplicates` too i.e. `df.tfidf.drop_duplicates("address", threshold=0.6, ngram=3)`


## Example

```python exec="yes" source="above" session="quickstart" result="python"
import liken as lk
import pandas as pd

df = pd.DataFrame(
    {
        "id": [1, 2],
        "email": [
            "fizzpop@yahoo.com",
            "FizzPop@yahoo.com"
        ],
    }
)

deduper = lk.dedupe(df).apply(lk.fuzzy(threshold=0.7))

print("No deduplication:")
print(df.drop_duplicates("email"))

# print("\nBut Liken does:")
print(df.fuzzy.drop_duplicates("email", threshold=0.7))

# TODO: this needs fixing!
print("\nUse the keep arg:")
print(df.fuzzy.drop_duplicates("email", threshold=0.7, keep="last"))
```

## Limitations

Pandas affordances support any deduper that works at a [threshold](../in-practice/common-recipes.md#threshold-vs-predicate) (such as [fuzzy](../reference/liken.md#liken.fuzzy), [tfidf](../reference/liken.md#liken.tfidf), [lsh](../reference/liken.md#liken.lsh), [jaccard](../reference/liken.md#liken.jaccard), and [cosine](../reference/liken.md#liken.cosine)). [Predicate](../in-practice/common-recipes.md#threshold-vs-predicate) dedupers are **not** supported.
