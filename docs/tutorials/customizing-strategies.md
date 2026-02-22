---
title: Customizing Strategies
---

## Defining Your Own Strategies

**Liken** supports bringing your own strategy. Let's look at an example of a custom strategy and then understand the requirements for defining one yourself. Although **Liken** provides a [`str_len`](../reference/rules.md#liken.rules.str_len) discrete strategy via it's `rules`sub-package, let's imagine our own, similar, implementation. Instead of bounding both the minimum and maximum length of the string, we'll require only a minimum length as well as equality of lengths between two records. We'll call it `str_same_len` and is implemented as:

```python
def str_same_len(array, *, min_len: int):
    n = len(array)
    for i in range(n):
        for j in range(i + 1, n):
            if len(array[i]) == len(array[j]) and len(array[i]) > min_len:
                yield i, j
```

Notice how the function has been written in such a way that it assumes no knowledge of an underlying DataFrame implementation. Instead, `array` is treated as generically representing an iterable collection of a DataFrame column. If you wanted to, you could pass in a pandas Series object into a similar looking function in order to retrieve similar records. 

There's a few other key take-aways to make:

- This is actually a *generator*, which is more efficiently consumed by **Liken**. You can choose to build a list and return that, but it's less efficient.
- Look at the signature: it requires the `min_len` argument to be passed as a *keyword*-argument. The general form required will have a signature that is denoted by `(array, *, **kwargs)`.
- The function is written in pure Python and is backend agnostic.


## Registering Custom Strategies

Having defined our custom strategy, we're now ready to make is useable by the `Dedupe` class. Right now our function won't ever know what `array` is and nor is it ready to plug into the inner workings of **Liken**. So, we're going to *register* the function for use:

```python {hl_lines="1 3"}
from liken.custom import register

@register
def str_same_len(array, *, min_len: int):
    n = len(array)
    for i in range(n):
        for j in range(i + 1, n):
            if len(array[i]) == len(array[j]) and len(array[i]) > min_len:
                yield i, j
```

Registering your function means it can be *applied* to `Dedupe` with the `apply()` function. Critically, the registering process means we can forget about `array` and how to represent it — `Dedupe` understands that it will have to construct `array` from the defined column(s) in your strategy, let's see what that looks like:

=== "Sequential API"

    ```python {hl_lines="2"}
    lk = Dedupe(df)
    lk.apply(str_same_len(min_len=12))
    df = lk.drop_duplicates("address")
    ```

=== "Dict API"

    ```python {hl_lines="2"}
    lk = Dedupe(df)
    lk.apply({"address": str_same_len(min_len=12)})
    df = lk.drop_duplicates()
    ```

=== "Rules API"

    ```python {hl_lines="2"}
    lk = Dedupe(df)
    lk.apply(Rules("address", on(str_same_len(min_len=12))))
    df = lk.drop_duplicates()
    ```

See how `array` isn't passed? `Dedupe` will retrieve an array representation of the `address` column in your DataFrame by itself, ensuring that the end usage of your custom strategy is as clean as simple as the rest of **Liken's** strategies.

!!! note
    Custom strategies can be combined using `&` in the **Rules API**. However, the negated form of your custom function will not be available using the `~` operator. Define an additional custom strategy in order to achieved that, for example, in the above case a `not_str_same_len` function.

!!! warning
    Currently only custom single column strategies are supported.

## Recap
!!! success "You learnt:"
    - Custom strategies can be defined using pure Python functions that have the signature `(array, *, **kwargs)`.
    - The function can return a list, or generate values on the fly. Generators are preferred.
    - You can make use of your function by using the `@register` decorator from the `liken.custom` sub-package.
    - You can apply your strategy just like you would any other **Liken** strategy. 