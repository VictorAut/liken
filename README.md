A Python library for grouping duplicate data efficiently.

<p align="center">
<a href="https://pypi.python.org/pypi/dupegrouper"><img height="20" alt="PyPI Version" src="https://img.shields.io/pypi/v/dupegrouper"></a>
<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/dupegrouper">
</p>

# Introduction

**dupegrouper** can be used for various deduplication use cases. It's intended purpose is to implement a uniform API that allows for both exact *and* near deduplication — whilst also offering record selection, based on the first instance of a set of duplicates i.e. a "group". 

Deduplicating data is a hard task — validating approaches takes time, can require a lot of testing, validating, and iterating through approaches that may, or may not, be applicable to your dataset.

**dupegrouper** abstracts away the task of *actually* deduplicating, so that you can focus on the most important thing: implementing an appropriate "strategy" to achieve your stated end goal ...

...In fact a "strategy" is key to **dupegrouper's** API. **dupegrouper** has:

### Ready-to-use deduplication strategies
**dupegrouper** currently offers the following deduplication strategies:
| string type | numeric type|
|:------------|-------------|
| Exact string       | Jaccard*    |
| Fuzzy matching       | Cosine similarity*    |
| TfIdf       | -    |
| LSH*       | -    |

\* *due for implementation in a future version*

You can also implement custom deduplication logic, which **dupegrouper** can readily accept, as descripted in [*Custom Strategies*](#custom-strategies).

### Multiple backend support
**dupegrouper** aims to scale in line with your problem. The following backends are currently support:
- Pandas
- Polars
- PySpark


### A flexible API

Checkout the [API Documentation](https://victorautonell-oiry.me/dupegrouper/dupegrouper.html)


## Installation


```shell
pip install dupegrouper
```

## Example

```python
import dupegrouper

dg = dupegrouper.DupeGrouper(df) # input dataframe

dg.add_strategy(dupegrouper.strategies.Exact())

dg.dedupe("address")

dg.df # retrieve dataframe
```

# Usage Guide

## Adding Strategies

**dupegrouper** comes with ready-to-use deduplication strategies:
- `dupegrouper.strategies.Exact`
- `dupegrouper.strategies.Fuzzy`
- `dupegrouper.strategies.TfIdf`


Strategies can be added one-by-one and are executed in *the order in which they are added*. In the below case the, the `address` column will firstly be deduplicted exactly, and then using Fuzzy matching.

```python
# Deduplicate the address column

dg = dupegrouper.DupeGrouper(df)

dg.add_strategy(dupegrouper.strategies.Exact())
dg.add_strategy(dupegrouper.strategies.Fuzzy(tolerance=0.3))

dg.dedupe("address")
```

Or, you can add a map of strategies. In this case, strategies are executed in their defined order, for each map key. The below implementation will produce the same as above.

```python
# Also deduplicates the address column

dg = dupegrouper.DupeGrouper(df)

dg.add_strategy({
    "address": [
        dupegrouper.strategies.Exact(),
        dupegrouper.strategies.Fuzzy(tolerance=0.3),
    ]
})

print(dg.strategies)
# {'address': ('Exact', 'Fuzzy', 'TfIdf')}

dg.dedupe() # No Argument!
```
A call of `dedupe()` will reset the strategies:
```bash
...
print(dg.strategies)
# {'address': ('Exact', 'Fuzzy', 'TfIdf')}
dg.dedupe()
print(dg.strategies)
# None
```

## Custom Strategies

Maybe you need some custom deduplication methodology. An instance of `dupegrouper.DupeGrouper` can accept custom functions too.

```python
def my_func(df, attr: str, /, **kwargs) -> dict[str, str]:
    my_map = {}
    for row in df:
        # e.g. use **kwargs
        my_map = ...
    return my_map
```

Above, `my_func` is a (very) boilerplate custom deduplication implementation: 
- it accepts a dataframe (`df`)
- it will deduplicate on a specific attribute (`attr`)
- And accepts other keyword arguments specific to your problem (`**kwargs`)

Look closely at the function signature — your function needs to implement this exactly. Additionally it produces a map where a key-value pair represents a deduplication match where the value is the new selected record ("group").

> [!WARNING]
> In the current implementation, there is no guarantee that a generator can be used to `yield` deduplicated value maps

You can proceed to add your custom function as a strategy:

```python
dg = dupegrouper.DupeGrouper(df)

dg.add_strategy((my_func, {"match_str": "london"}))

print(dg.strategies) # returns ("my_func",)

dg.dedupe("address")
```


> [!WARNING]
> In the current implementation, any custom callable will also *always dedupe exact matches!*

## Creating a Comprehensive Strategy

You can use the above techniques for a comprehensive strategy to deduplicate your data:

```python
import dupegrouper
import pandas # | polars | pyspark

df = pd.read_csv("example.csv")

dg = dupegrouper.DupeGrouper(df)

strategies = {
    "address": [
        dupegrouper.strategies.Exact(),
        dupegrouper.strategies.Fuzzy(tolerance=0.5),
        (my_func, {"match": "london"}), # any address that contains "london"
    ],
    "email": [
        dupegrouper.strategies.Exact(),
        dupegrouper.strategies.Fuzzy(tolerance=0.3),
        dupegrouper.strategies.TfIdf(tolerance=0.4, ngram=3, topn=2),
    ],
}

dg.add_strategy(strategies)

dg.dedupe()

df = dg.df
```

## Using the PySpark backend

**dupegrouper** can be used as described in [*Creating a Comprehensive Strategy*](#creating-a-comprehensive-strategy). **dupegrouper** is partition-aware, and will deduplicate each partition, per worker node, given the defined strategy. Such a distributed implementation puts the onus on appropriate planning:
- partitions must *already* be containers of *expected* duplicates
- partitioning (or *re*-partitioning) must be planned ahead of time

The above problem is typically dealt with the use of a "blocking key" which is the partitioning/repartitioning key. Whilst several approaches may be valid, a blocking key be typically be computed as a general property of several records that are expected to contain duplicates. As an example, that might be the first *N* characters of a given attribute needing deduplicating.

## Extending the API for Custom Implementations
It's recommended that for simple custom implementations you use the approach discussed for custom functions. (see [*Custom Strategies*](#custom-strategies)).

However, you can derive directly from the abstract base class `dupegrouper.strategy.DeduplicationStrategy`, and thus make direct use of the efficient, core deduplication methods implemented in this library, as described in it's [API](./dupegrouper/strategy.html#DeduplicationStrategy). This will expose a `dedupe()` method, ready for direct use within an instance of `DupeGrouper`, much the same way that other `dupegrouper.strategies` are passed in as strategies.

# About

## License
This project is licensed under the [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0.html). See the [LICENSE](LICENSE) file for more details.