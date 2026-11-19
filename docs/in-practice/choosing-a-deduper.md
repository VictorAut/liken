---
title: Choosing a Deduper
---

## Built-in Dedupers

Liken comes with many deduplication methods built-in:

| |               | Deduper                                              | Description                                                                 | Complexity |
|-------------| ------------- | ----------------------------------------------------- | ---------------------------------------------------------------- | -------- |
| *Threshold* |*single-column*| [`exact`](../reference/liken/#liken.exact)       | Exact deduplication.  | *O(n)* |
| *Threshold* |*single-column*| [`fuzzy`](../reference/liken/#liken.fuzzy)       | Fuzzy string matching.                                                                            | *O(n²)* |
| *Threshold* |*single-column*| [`edit_distance`](../reference/liken/#liken.edit_distance) | Values within a maximum edit (Levenshtein) distance. | *O(n²)* |
| *Threshold* |*single-column*| [`within`](../reference/liken/#liken.within)     | Values within an absolute numeric tolerance. | *O(n log n)* |
| *Threshold* |*single-column*| [`tfidf`](../reference/liken/#liken.tfidf)       | String token matching with Tf-Idf. | *~O(n·c)* |
| *Threshold* |*single-column*| [`lsh`](../reference/liken/#liken.lsh)           | String token matching with Locality Sensitive Hashing (LSH). | *O(n)* |
| *Threshold* |*single-column*| [`sorted_neighbourhood`](../reference/liken/#liken.sorted_neighbourhood) | Fuzzy string matching inside a window of sorted-order neighbours. | *O(n log n)* |
| *Threshold* |*compound-column*| [`jaccard`](../reference/liken/#liken.jaccard) | Multi column similarity based on intersection of categorical data.                                | *O(n²)* |
| *Threshold* |*compound-column*| [`cosine`](../reference/liken/#liken.cosine)   | Multi column similarity based on dot product of numerical data.                                   | *O(n²)* |
| *Threshold* |*compound-column*| [`token_overlap`](../reference/liken/#liken.token_overlap) | Multi column set containment: a record whose values are a subset of another's matches.        | *O(n²)* |
| *Threshold* |*compound-column*| [`knn`](../reference/liken/#liken.knn) | Multi column k nearest neighbours on numerical data: each record matches its k best under a similarity threshold.        | *O(n log n)* |
| *Predicate* |*single-column*| [`isna`](../reference/liken/#liken.isna)                | Records where the column value is missing (`None`/`NaN`).                             | *O(n)* |
| *Predicate* |*single-column*| [`isin`](../reference/liken/#liken.isin)                | Records where the column value is in a list of members.                               | *O(n)* |
| *Predicate* |*single-column*| [`str_startswith`](../reference/liken/#liken.str_startswith)     | Records where the string starts with a pattern. | *O(n)* |
| *Predicate* |*single-column*| [`str_endswith`](../reference/liken/#liken.str_endswith)       | Records where the string ends with a pattern. | *O(n)* |
| *Predicate* |*single-column*| [`str_contains`](../reference/liken/#liken.str_contains)         | Records where the string contains a pattern. Accepts Regex. | *O(n)* |
| *Predicate* |*single-column*| [`str_len`](../reference/liken/#liken.str_len) | Records where the string length is bounded by a minimum and maximum length. | *O(n)* |
| *Predicate* |*single-column*| [`num_range`](../reference/liken/#liken.num_range) | Records where the numeric value falls inside the inclusive interval. | *O(n)* |
| *Predicate* |*single-column*| [`date_range`](../reference/liken/#liken.date_range) | Records whose dates, parsed under an explicit format, are at most `window` days apart. | *O(n)* |
| *Predicate* |*single-column*| [`sorted_tokens`](../reference/liken/#liken.sorted_tokens) | Records whose whitespace-separated words are the same, in any order. | *O(n)* |

*Single-column* dedupers apply to single columns and are implementation of near string matching. *Compound-column* dedupers are set operations where the values of the set are the values of the columns in a given record. *Threshold* dedupers have a `threshold` argument. *Predicate* dedupers choose an outcome based on a discrete outcome (e.g. missing / not missing).

To *use* dedupers, you have to *apply* them, which is covered in the next tutorial.

## Choosing a Deduper

One route from your data problem to a built-in deduper. Start at the top and answer for the column you want to deduplicate:

- *Teal* dedupers are threshold dedupers
- *Orange* dedupers are predicate dedupers

``` mermaid
flowchart TD

    classDef predicate fill:#fde68a,stroke:#b45309,color:#78350f
    classDef similarity fill:#99f6e4,stroke:#0d9488,color:#134e4a

    q{{"`**What are you deduplicating?**`"}}

    single(["`**One column**`"])
    sstr(["`**String**`"])
    num(["`**Numeric**`"])
    date(["`**Date**`"])
    comp(["`**Several Columns**`"])

    q --> single
    q --> comp

    single -- Membership or missing checks on any type --> anypred
    anypred["`**isin** <br/>**isna**`"]

    single --> sstr
    single --> date
    single --> num

    sstr -- Exact copies --> sexact
    sexact["`**exact**`"]
    sstr -- Same words, in any order --> stok
    stok["`**sorted_tokens**`"]
    sstr -- String predicates --> spred
    spred["`**str_startswith** <br/>**str_endswith** <br/>**str_contains** <br/>**str_len**`"]
    sstr --Typos or small edits --> sfuzzy
    sfuzzy["`**fuzzy** <br/>**edit_distance** <br/>**sorted_neighbourhood**`"]
    sstr -- Large columns, token similarity --> stfidf
    stfidf["`**lsh** <br/>**tfidf**`"]


    date -- Date in range --> ndate
    ndate["`**date_range**`"]

    num -- Within an absolute tolerance --> nwithin
    num -- Value in range --> nrange
    nwithin["`**within**`"]
    nrange["`**num_range**`"]

    comp -- Categorical sets --> cjac
    cjac["`**jaccard** <br/>**token_overlap**`"]
    comp -- Numeric vectors --> cvec
    cvec["`**cosine** <br/>**knn**`"]

    nrange ~~~ sstr

    class cjac,cvec,sed,nwithin,sexact,stok,sfuzzy,stfidf,ssn similarity
    class anypred,nrange,ndate,spred predicate
```


## Choosing a Threshold

Threshold dedupers compare values pairwise and keep the pairs whose score meets their `threshold`. All thresholds are on a 0–1 scale, and matching is inclusive: a pair scoring exactly *at* the threshold matches. At the extreme, `threshold=0` matches every pair, including pairs with no similarity at all. Thresholds differ in what they measure:

- `fuzzy` uses [`rapidfuzz`](https://rapidfuzz.github.io/RapidFuzz/) string similarity. The default scorer, `simple_ratio`, compares whole strings character by character: `"london"` and `"londn"` score about 0.91. Matching is case-sensitive: `"LONDON"` and `"london"` score 0.0. Other scorers (`token_sort_ratio`, `token_set_ratio`, and more) change how strings are compared, not the scale. With `topn=k`, `fuzzy` becomes a blocking strategy: each value keeps only its `k` best candidate pairs (a pair matches when either side holds the other), which bounds the scores held per value, but a cluster of more than `k` near-identical records may not link.
- `tfidf` splits each value into character n-grams (`ngram`, default 3), weights them, and compares the resulting vectors by cosine similarity. It tolerates more drift than `fuzzy`, but its default `topn=2` is a blocking strategy too: it keeps only the two best candidate pairs per record, one of which is the record itself. In dense clusters of near-duplicates, raise `topn`.
- `lsh` builds MinHash signatures over character n-grams and matches approximately: the threshold estimates the Jaccard similarity of the n-gram sets. LSH is a blocking strategy — it scales well to very large data, but a match is probabilistic — pairs near the threshold can be missed.
- `sorted_neighbourhood` is also a blocking strategy: it sorts the column by value and scores each record only against the next `window - 1` records in sorted order, so scorer work falls from O(n²) to O(n·window). The match set is a subset of all-pairs `fuzzy` at the same threshold and scorer. Sorting is on raw values: similar values differing near the start ("OL5 9PL" vs "0L5 9PL") can fall outside each other's windows, and word-order variants are `sorted_tokens`' job. `fuzzy(topn=k)` is the different trade-off: k best partners anywhere in the column, still all-pairs scoring.

`explore` shows where your data's duplicate rates actually fall, so you can pick a threshold from evidence rather than guesswork.

## Handling Missing Values

Liken does not drop missing values and treats any missing value as `None` or any `NaN`.

| Deduper | What happens to missing values |
| --- | --- |
| `exact` | Cluster with each other|
| `fuzzy` | Cluster with each other|
| `edit_distance` | Cluster with each other|
| `within` | Cluster with each other|
| `tfidf` | Cluster with each other at any given `ngram`; the vectorizer receives non-missing values only. |
| `lsh` | Same as `tfidf`. |
| `isna` | Matches exactly the missing values; `~isna` matches exactly the non-missing values. |
| `num_range` | Never matches: a missing value is outside any interval, positive or negated (`~num_range`). |
| `date_range` | Never matches: a missing value parses to nothing, positive or negated (`~date_range`). |
| `sorted_tokens` | Never matches: a missing value has no words to sort, positive or negated (`~sorted_tokens`). |
| Other predicates | A missing value never satisfies a predicate, except positive `isin`, where a missing value matches only if `None` is listed in `values`. |
