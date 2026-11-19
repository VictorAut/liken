---
Titls: Representative Records
---


## Synthetizing Representative Records

A canonical record can be linked to several child records. Use `.synthesize` to create a new record per canonical cluster, coalescing the values of the cluster's child records:

```python
import liken as lk

result = (
    lk.dedupe(df)
    .apply(lk.fuzzy(threshold=0.85))
    .canonicalize(
        "email",
        keep="first",
        source_id="uid", # `source_id` arg included
    )
)

synthetic_records = result.synthesize()
```

`synthesize` returns a new DataFrame with exactly one record per canonical cluster. For each column, the synthetic value is the *first non-missing* value among the cluster's records, in row order. The canonicalized DataFrame itself keeps all child records, unchanged — the synthetic records sit alongside them, in their own frame:

 canonical_id  |  uid   |  address  |        email
---------------|--------|-----------|---------------------
      a001     |  a001  |  london   |   fizzpop@yahoo.com
      a003     |  a003  |   paris   |       a@msn.fr

/// caption
One synthetic record per canonical cluster: `a001`'s cluster coalesces into a single record.
///

First non-missing means the first record's missing value does not win — the first *non-missing* value does:

  uid   |  address  |  email
--------|-----------|----------
  b001  |   None    |  x@y.io
  b002  |   oslo    |  x@y.io

```python
import liken as lk

result = lk.dedupe(null_df).apply(lk.exact()).canonicalize("email", source_id="uid")
result.synthesize()
```

 canonical_id  |  uid   |  address  |  email
---------------|--------|-----------|----------
      b001     |  b001  |   oslo    |  x@y.io

/// caption
`b001`'s address is missing, so the synthetic record takes `b002`'s.
///

`synthesize` collects to your machine on distributed backends. Unlike `canonicals`, it does not refuse to run before a `canonicalize` — on an un-canonicalized chain it silently treats every record as its own cluster, which is rarely what you want.
