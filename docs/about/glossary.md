---
title: Glossary
---

The vocabulary of liken's documentation. Write against these terms.

### Blocking

Literature synonym of [candidate generation](#candidate-generation).
Liken writes "candidate-generation (blocking) strategy" — see the
[dedupers](../reference/liken.md) and the tutorials.

### Candidate generation

The step that reduces all possible record pairs to a manageable set of
[candidate pairs](#candidate-pairs). Liken calls the mechanisms that do
this *candidate-generation (blocking) strategies*: `topn`, `lsh`,
`sorted_neighbourhood`.

### Candidate pairs

The pairs of records a deduper produces after candidate generation.
Only candidate pairs are compared; the rest are never scored.

### Canonicalize

In liken, the pipeline stage that executes the applied dedupers and
labels duplicate records with a shared `canonical_id`. Every record is
kept, and the record chosen by survivorship is the cluster's
[representative](#representative-record). A synthetic consolidated
record can be built separately with `synthesize`.

In ER literature, canonicalization usually means consolidation:
merging each [cluster](#clustering) of duplicates into a single record.
Liken does not merge records at this stage.

In prose, the en-GB `-ise` spelling of this word is retired; the API
and the docs use `canonicalize`.

### Clustering

The step that links candidate pairs into clusters. Matching is
transitive — see [transitive closure](#transitive-closure) — so a
cluster is a chain of matches, not a set of records that all match each
other directly.

### Deduplication

Finding records that describe the same thing and acting on them —
removing or merging duplicates. A *deduper* defines what counts as a
duplicate; it produces [candidate pairs](#candidate-pairs) and links
them into [clusters](#clustering).

### Entity resolution

Another name for [record linkage](#record-linkage) in the literature.
Liken's documentation uses *record linkage*.

### Golden record

Retired. Not liken vocabulary. Use
[representative record](#representative-record).

### Match rule

Literature synonym of a *deduper* — the definition of what counts as a
duplicate. Liken writes *deduper*.

### Record linkage

Labelling records that describe the same entity instead of dropping
them: linked records are kept and receive a shared `canonical_id`. See
the [Record Linkage tutorial](../tutorials/record-linkage.md).

### Representative record

The record chosen by the [survivorship](#survivorship) policy to stand
for its [cluster](#clustering).

### Survivorship

The policy that picks each cluster's
[representative record](#representative-record): `"first"` keeps the
earliest record of the cluster in row order, `"last"` the latest.

### Transitive closure

Matching records through chains. If record `a` matches `b`, and `b`
matches `c`, all three belong to one [cluster](#clustering) — even when
`a` and `c` would not match each other directly.
