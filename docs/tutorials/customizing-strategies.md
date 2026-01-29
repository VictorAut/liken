---
title: Customizing Strategies
---

## Defining Your Own Strategies

**Enlace** supports bringing your own strategy. Let's look at an example of a custom strategy and then understand the requirements for defining one yourself. Although **Enlace** provides a [`str_len`](../../../reference/rules/#enlace.rules.str_len) discrete strategy via it's `rules`sub-package, let's imagine our own, similar, implementation that instead of bounding the required lengths instead deduplicates any two records that have the same length, given a minimum length. We'll call it `str_same_len` and is implemented as:

```python
def str_same_len(array, *, min_len: int):
    n = len(array)
    for i in range(n):
        for j in range(i + 1, n):
            if len(array[i]) == len(array[j]) and len(array[i]) > min_len:
                yield i, j
```

Notice how the function has been written in such a way that it assumes no knowledge of an underlying DataFrame implementation. Instead, `array` is treated as generically representing an iterable collection of a DataFrame column. If you wanted to you could pass in a pandas Series object into a similar looking function in order to retrieve similar records. 

There's a few other key take-aways to make:

- This is actually a *generator*, which is more efficiently consumed by **Enlace**. You can choose to build a list, too, but it's less efficient.
- Look at the signature: it requires the `min_len` argument to be passed as a *keyword*-argument. The general form required will have a signature that is deonted by `(array, *, **kwargs)`.
- The function is written in pure Python and is backend agnostic.


## Registering Custom Strategies

Having defined our custom strategy, we're now ready to make is useable by the `Dedupe` class. Right now our function won't ever know what `array` is and nor is ready to plug into the inner workings of **Enlace**. We're going to *register* the function for use:

```python {hl_lines="1 3"}
from enlace.custom import register

@register
def str_same_len(array, *, min_len: int):
    n = len(array)
    for i in range(n):
        for j in range(i + 1, n):
            if len(array[i]) == len(array[j]) and len(array[i]) > min_len:
                yield i, j
```

Now that you've registered your function it can be *applied* and critically the registration process means we can forget about `array` — `Dedupe` understands that it will have to construct `array` from a the defined columns in your strategy, let's see what that looks like when used with Sequential API:

```python {hl_lines="2 5"}
dp = Dedupe(df)
dp.apply(str_same_len(min_len=12))
df = dp.drop_duplicates("address")
```

See how `array` isn't passed? `Dedupe` will retrieve an array representation of the `address` column in your DataFrame by itself, ensuring that the end usage of your custom strategy is as clean as simple as the rest of **Enlace's** strategies.

!!! note
    You can use any of the APIs when using custom strategies. You can also access combinations by using the Rules API. However, you will have to make *another* custom implementation for any negated form of your function.

## Recap
!!! success "You learnt:"
    - Custom strategies can be defined using pure Python where the function has the following signature `(array, *, **kwargs)`
    - The function can return a list, or generate values on the fly. Generators are preferred.
    - You can make use of your function by using the `@register` decorator
    - Finally, you can apply your strategy just like you would any other **Enlace** strategy. 