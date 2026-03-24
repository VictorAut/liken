---
title: Custom Dedupers
---

**Liken** supports defining your own, custom, dedupers.

??? warning "Limitations"
    **Liken** currently only guarantees usage of custom *single-column* dedupers.

## Defining a Custom Deduper

Custom dedupers are defined as Python functions. There are strict requirements to the shape of your function — it must assume no knowledge of your DataFrame, and instead operate on a generic, unlabelled "array", where the "array" must be iterable and and an abstraction of a column in your dataset. Let's look at an example.

Although **Liken** provides a [`str_len`](../reference/liken.md#liken.str_len) predicate deduper, we'll define our own, similar, implementation: `str_same_len`. `str_same_len` will be defined deduplicate records that have a minimum length as well as equality of lengths between any two given records:

```python
def str_same_len(array, *, min_len: int):
    n = len(array)
    for i in range(n):
        for j in range(i + 1, n):
            if len(array[i]) == len(array[j]) and len(array[i]) > min_len:
                yield i, j
```

Notice how the function has been written in such a way that it assumes no knowledge of an underlying DataFrame implementation. Instead, `array` is treated as generically representing an iterable collection of a DataFrame column.

There's a few other key take-aways to make:

- This is actually a *generator*, which is more efficiently consumed by **Liken**. You can choose to build a list and return that instead, but it's less efficient.
- Look at the signature: it requires the `min_len` argument to be passed as a *keyword*-argument. The general form required will have a signature that is denoted by `(array, *, **kwargs)`.
- The function is written in pure Python and is backend agnostic.


## Using a Custom Deduper

A defined deduper needs to be "registered" to be made available to **Liken**:

```python {hl_lines="3"}
import liken as lk

@lk.custom.register
def str_same_len(array, *, min_len: int):
    n = len(array)
    for i in range(n):
        for j in range(i + 1, n):
            if len(array[i]) == len(array[j]) and len(array[i]) > min_len:
                yield i, j
```

Registering your function means it can be *applied* with the `apply()` function. Registering process means we can forget about `array` and how to represent it — **Liken** will construct `array` from the defined column(s) in your deduper:

=== "Single deduper"

    ```python
    df = (
        lk.dedupe(df)
        .apply(str_same_len(min_len=12))
        .drop_duplicates("address")
        .collect()
    )
    ```

=== "Dict collection"

    ```python
    df = (
        lk.dedupe(df)
        .apply({"address": str_same_len(min_len=12)})
        .drop_duplicates()
        .collect()
    )
    ```

=== "Pipeline"

    ```python
    df = (
        lk.dedupe(df)
        .apply(lk.pipeline().step(lk.col("address").str_same_len(min_len=12)))
        .drop_duplicates()
        .collect()
    )
    ```

Note how `array` isn't passed as an argument in any instance. `dedupe` will retrieve an array representation of the `address` column, ensuring that usage of the custom deduper matches that of other **Liken** dedupers.

!!! note
    Custom dedupers can be combined using AND semantics in pipelines. However, the negated form of your custom function will not be available using the `~` operator. Define an additional custom deduper in order to achieved that. In the above case a `not_str_same_len` function, say.