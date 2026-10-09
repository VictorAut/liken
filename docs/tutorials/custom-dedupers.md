---
title: Custom Dedupers
---

Liken supports defining your own, custom, dedupers.

Liken currently only guarantees usage of custom *single-column* dedupers. "Bringing your own deduper" comes with other [limitations](#limitations)

## The Callable Contract

A custom deduper is a plain Python function with a strict shape:

- It receives only **one positional argument**, `array`: a *Collection* of a column's values, in DataFrame row order.
- An arbitrary number of keyword arguments defining parameters of your function are accepted.
- From this list it must return or yield pairs of positional indices (i.e. two-element tuples `(i, j)`) naming the records needing linkage. Indices are 0-based positions into the list as handed to the function, which is the DataFrame's row order.

## Defining a Custom Deduper

Your function will need to be *registered* for usage with Liken.

### Worked Example

Although Liken provides a [`str_len`](../reference/liken.md#liken.str_len) predicate deduper, we'll define our own, similar, implementation: `str_same_len`. `str_same_len` will deduplicate records whose values share a length, as long as that length is at or above a minimum:

```python {hl_lines="3"}
import liken as lk

@lk.custom.register
def str_same_len(array, *, min_len: int):
    n = len(array)
    for i in range(n):
        for j in range(i + 1, n):
            if len(array[i]) == len(array[j]) and len(array[i]) >= min_len:
                yield i, j
```

See it work on a small input dataset:

 id   |     address
------|------------------
  0   |  10 high street
  1   |  99 high street
  2   |  22 acacia avenue
  3   |  5 low road

/// caption
Lengths are 14, 14, 16 and 10. With `min_len=12`, rows 0 and 1 share a length at or above the minimum; row 2 has no equal-length partner; row 3 is excluded outright.
///

Applying it with `min_len=12`:

```python
import liken as lk

df = (
    lk.dedupe(df)
    .apply(str_same_len(min_len=12)) # (1)!
    .drop_duplicates("address")
)
```

1. Note that your custom function is implicitely passed an `array`, which Liken picks up as "address"; and therefore you *only* need to pass additional `kwargs`!


 id   |     address
------|------------------
  0   |  10 high street
  2   |  22 acacia avenue
  3   |  5 low road

/// caption
Rows 0 and 1 were linked and collapsed; the rest survive.
///

## Using a Custom Deduper

`lk.custom.register` wraps your function in a deduper object and registers it under the function's name. Registration is what lets you forget about `array` — Liken constructs it from the column your deduper is applied to:

=== "Single deduper"

    ```python
    df = (
        lk.dedupe(df)
        .apply(str_same_len(min_len=12))
        .drop_duplicates("address")
    )
    ```

=== "Dict collection"

    ```python
    df = (
        lk.dedupe(df)
        .apply({"address": str_same_len(min_len=12)})
        .drop_duplicates()
    )
    ```

=== "Pipeline"

    ```python
    df = (
        lk.dedupe(df)
        .apply(lk.pipeline().step(lk.col("address").str_same_len(min_len=12)))
        .drop_duplicates()
    )
    ```

Note how `array` isn't passed as an argument in any instance. `dedupe` will retrieve an array representation of the `address` column, ensuring that usage of the custom deduper matches that of other Liken dedupers.

The keyword arguments you pass at the call site (`min_len=12` above) are stored on the registered deduper and forwarded to your function only when deduplication runs. The registered name also becomes available as a method on the `lk.col(...)` expression, so pipelines treat it like any built-in deduper (as the Pipeline tab shows). Registering a name that a built-in already uses shadows the built-in *on the `lk.col(...)` expression*, so ensure you choose distinct names.

## Predicate-Style Results

A custom deduper yields pairs, but nothing stops you yielding the pairs that a *predicate* deduper would. A predicate groups **all matching records into one group**:

```python
import liken as lk

@lk.custom.register
def same_domain(array, *, suffix: str):
    matches = [i for i, v in enumerate(array) if str(v).endswith(suffix)]
    for i in matches[1:]:
        yield matches[0], i
```

Applied to four email addresses:

```python
import liken as lk

result = (
    lk.dedupe(email_df)
    .apply(lk.pipeline().step(lk.col("email").same_domain(suffix="@gmail.com")))
    .canonicalize()
)
```

`same_domain(suffix="@gmail.com")` links the two Gmail records and leaves the rest untouched:

 id   |   canonical_id   |     email
------|------------------|----------------
  0   |        0         |  a@gmail.com
  1   |        1         |  b@ex.io
  2   |        0         |  c@gmail.com
  3   |        3         |  d@ex.io

/// caption
The two `@gmail.com` records share a canonical id; the others keep their own.
///

??? note "Inversion of a custom, predicate, deduper"
    Your custom deduper is still treated as a threshold deduper, not a predicate: you cannot invert it with `~`, and rule predication does not single it out. It can however be combined with real predicate dedupers in an AND step. See the limitations below.

## Limitations

- **Only single-column custom dedupers are guaranteed.** A custom function may declare a tuple of columns, and it will receive a list of dicts (one per record) but that shape is not a tested guarantee.
- **`~` negation is unavailable.** Negation is defined for *predicate* dedupers only, and a registered custom deduper is a threshold deduper regardless of what it yields. Applying `~` raises `TypeError: Only predicate dedupers support inversion`. To get the negated behaviour, define a second custom function — a `not_str_same_len`, say, that yields the complementary pairs.

Custom dedupers **can** be combined using AND semantics in pipelines with other dedupers.

## Data Size and Distributed Backends

Your function always receives a plain Python list, which Liken produces from the column's in-memory representation. That has consequences for size and placement:

- The list holds one Python object per value, so memory use is proportional to the column size.
- On local backends (pandas, polars, modin, pyarrow) the list covers the whole column.
- On distributed backends (dask, ray, pyspark) Liken runs your function per partition, per batch; on ray, the worker holding it. The DataFrame is not pulled to one machine, but instead each slice's column is materialised as a list on a single worker, and deduplication matches records only within that slice.
- Your function must survive being sent to workers: keep it importable and picklable, and avoid closures over unserialisable state.
