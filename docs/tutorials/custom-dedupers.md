---
title: Custom Dedupers
---

**Liken** supports defining your own, custom, dedupers.

**Liken** currently only guarantees usage of custom *single-column* dedupers. The reasons are covered in [Limitations](#limitations), along with the missing `~` negation.

## The Callable Contract

A custom deduper is a plain Python function with a strict shape. It receives **one positional argument**: a plain Python list of the column's values, in DataFrame row order. From this list it must yield pairs of positional indices — two-element tuples `(i, j)` — naming the records needing linkage. Indices are 0-based positions into the list as handed to the function, which is the DataFrame's row order.

```python
def str_same_len(array, *, min_len: int):
    n = len(array)
    for i in range(n):
        for j in range(i + 1, n):
            if len(array[i]) == len(array[j]) and len(array[i]) > min_len:
                yield i, j
```

The properties of the contract:

- The function sees **no DataFrame** — no column labels, no dtypes, no backend. `array` is a generic iterable of values. This is what makes the function backend-agnostic.
- Each yielded pair links two records. **Liken** unions the pairs: chain `(0, 1)` and `(1, 2)` and all three records land in one group.
- The function is a *generator*, which **Liken** consumes pair by pair without materialising the result. You can build a list and return that instead but holds every pair in memory at once, and is not recommended.
- Every further argument must be keyword-only (`(array, *, **kwargs)`). Positional call-site arguments are rejected.

## Defining a Custom Deduper

Although **Liken** provides a [`str_len`](../reference/liken.md#liken.str_len) predicate deduper, we'll define our own, similar, implementation: `str_same_len`. `str_same_len` will deduplicate records whose values share a length, as long as that length is above a minimum. The function from the contract above, registered:

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

See it work on a small dataset:

 id   |     address
------|------------------
  0   |  10 high street
  1   |  99 high street
  2   |  22 acacia avenue
  3   |  5 low road

/// caption
Lengths are 14, 14, 16 and 10. With `min_len=12`, rows 0 and 1 share a length above the minimum; row 2 has no equal-length partner; row 3 is excluded outright.
///

Applying it with `min_len=12`:

```python
import liken as lk

df = lk.dedupe(df).apply(str_same_len(min_len=12)).drop_duplicates("address")
```

 id   |     address
------|------------------
  0   |  10 high street
  2   |  22 acacia avenue
  3   |  5 low road

/// caption
Rows 0 and 1 were linked and collapsed; the rest survive.
///

## Using a Custom Deduper

`lk.custom.register` wraps your function in a deduper object and registers it under the function's name. Registration is what lets you forget about `array` — **Liken** constructs it from the column(s) your deduper is applied to:

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

Note how `array` isn't passed as an argument in any instance. `dedupe` will retrieve an array representation of the `address` column, ensuring that usage of the custom deduper matches that of other **Liken** dedupers.

The keyword arguments you pass at the call site (`min_len=12` above) are stored on the registered deduper and forwarded to your function only when deduplication runs. The registered name also becomes available as a method on the `lk.col(...)` expression, so pipelines treat it like any built-in deduper (as the Pipeline tab shows). Registering a name that a built-in already uses shadows the built-in *on the `lk.col(...)` expression*, so ensure you choose distinct names.

## Predicate-Style Results

A custom deduper yields pairs, but nothing stops you yielding the pairs that a *predicate* deduper would. A predicate groups **all matching records into one group** — so yield a pair from the first matching record to every other matching record:

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

!!! note
    Your custom deduper is still treated as a threshold deduper, not a predicate: you cannot invert it with `~`, and rule predication does not single it out. It can however be combined with real predicate dedupers in an AND step.

## Limitations

Two limitations apply to custom dedupers, and both have reasons:

- **Only single-column custom dedupers are guaranteed.** A custom function may declare a tuple of columns, and it will receive a list of dicts (one per record) but that shape is not a tested guarantee. The single-column shape is: nulls are substituted with the literal string `"na"` before your function sees them, consistently across backends. Compound columns receive raw values, nulls included, with no substitution.
- **`~` negation is unavailable.** Negation is defined for *predicate* dedupers only, and a registered custom deduper is a threshold deduper regardless of what it yields. Applying `~` raises `TypeError: Only predicate dedupers support inversion`. To get the negated behaviour, define a second custom function — a `not_str_same_len`, say, that yields the complementary pairs.

Custom dedupers **can** be combined using AND semantics in pipelines with other dedupers.

## Data Size and Distributed Backends

Your function always receives a plain Python list, which **Liken** produces from the column's in-memory representation. That has consequences for size and placement:

- The list holds one Python object per value, so memory use is proportional to the column, a cost your function pays before it yields anything.
- On local backends (pandas, polars, modin) the list covers the whole column.
- On distributed backends (dask, ray, pyspark) **Liken** runs your function per partition, per batch; on ray, the worker holding it. The DataFrame is not pulled to one machine, but instead each slice's column is materialised as a list on a single worker, and deduplication matches records only within that slice.
- Your function must survive being sent to workers: keep it importable and picklable, and avoid closures over unserialisable state.
