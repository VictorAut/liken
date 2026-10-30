"""Batched top-k pair generation for `fuzzy(topn=k)`.

WI070: `_topk_pairs` moves from per-row `process.extract` to the block
machinery (`process.cdist` strips, `resolve_workers`). The parity tests
pin the batched implementation's emitted pairs — set, order, and ties —
to a reference copy of the pre-WI070 per-row algorithm kept here.
"""

import numpy as np
import pyarrow as pa
from rapidfuzz import process

import liken as lk
from liken.dedupers.fuzzy import _BOUNDARY_TOLERANCE


def _reference_topk_pairs(scored, present, threshold, scorer, topn):
    """The pre-WI070 per-row `_topk_pairs` algorithm, kept as the parity
    reference.

    Copied verbatim from `fuzzy.py` at WI049's final state. The batched
    implementation must reproduce this pair sequence exactly.
    """
    cutoff: float = threshold - _BOUNDARY_TOLERANCE
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []

    for pos, s1 in enumerate(scored):
        results = process.extract(
            s1,
            scored,
            scorer=scorer,
            limit=topn + 1,
            score_cutoff=cutoff,
        )

        partners = 0
        for _choice, score, index in results:
            if partners == topn:
                break
            if index == pos:
                continue
            partners += 1
            if score < cutoff:
                continue
            pair = (min(present[pos], present[index]), max(present[pos], present[index]))
            if pair not in seen:
                seen.add(pair)
                out.append(pair)
    return out


def _assert_parity(array_values, threshold, topn, scorer_name):
    """The batched path must emit exactly the reference's pair sequence.

    The reference takes the threshold on the scorer's 0-100 scale (as
    `_gen_similarity_pairs` passes it): `100 * threshold` here.
    """
    scorer = lk.fuzzy(scorer=scorer_name).get_scorer()
    array = pa.array(array_values)
    present = list(range(len(array_values)))

    expected = _reference_topk_pairs(array_values, present, 100 * threshold, scorer, topn)
    got = list(lk.fuzzy(threshold=threshold, topn=topn, scorer=scorer_name)._gen_similarity_pairs(array))

    assert got == expected


def test_fuzzy_topk_blockwise_parity_graded_ratio():
    """Graded clusters (24-char stems, per-member mutations), native scorer."""
    values = []
    stems = ["kingstreet", "harborview", "millpond", "eastgate", "elmrow"]
    for c in range(30):
        for j in range(10):
            stem = stems[(c + j) % len(stems)]
            values.append(stem + ("x" * (j % 7)) + str(c * 31 + j))

    _assert_parity(values, 0.8, 5, "simple_ratio")


def test_fuzzy_topk_blockwise_parity_graded_jaro_winkler():
    """The jaro_winkler wrapper ignores the cdist cutoff; parity holds."""
    values = []
    stems = ["kingstreet", "harborview", "millpond", "eastgate", "elmrow"]
    for c in range(30):
        for j in range(10):
            stem = stems[(c + j) % len(stems)]
            values.append(stem + ("x" * (j % 7)) + str(c * 31 + j))

    _assert_parity(values, 0.8, 5, "jaro_winkler")


def test_fuzzy_topk_blockwise_parity_graded_partial_ratio():
    """partial_ratio's substring alignment admits more pairs; parity holds."""
    values = []
    stems = ["kingstreet", "harborview", "millpond", "eastgate", "elmrow"]
    for c in range(30):
        for j in range(10):
            stem = stems[(c + j) % len(stems)]
            values.append(stem + ("x" * (j % 7)) + str(c * 31 + j))

    _assert_parity(values, 0.8, 5, "partial_ratio")


def test_fuzzy_topk_blockwise_parity_dissimilar():
    """Long, mostly-dissimilar values at a high threshold: 6k+ pairs."""
    rng = np.random.default_rng(1234)
    words = ["alpha", "beta", "gamma", "delta", "omega", "zeta", "theta", "iota"]
    values = [f"{'-'.join(rng.choice(words, size=3))}-{rng.integers(0, 10**6):06d}" for _ in range(2000)]

    _assert_parity(values, 0.95, 5, "simple_ratio")


def test_fuzzy_topk_blockwise_parity_dissimilar_jaro_winkler():
    rng = np.random.default_rng(1234)
    words = ["alpha", "beta", "gamma", "delta", "omega", "zeta", "theta", "iota"]
    values = [f"{'-'.join(rng.choice(words, size=3))}-{rng.integers(0, 10**6):06d}" for _ in range(2000)]

    _assert_parity(values, 0.95, 5, "jaro_winkler")


def test_fuzzy_topk_blockwise_parity_all_ties():
    """All values identical: every score ties at 100, ties break by index."""
    _assert_parity(["identical"] * 300, 0.95, 5, "simple_ratio")


def test_fuzzy_topk_blockwise_parity_all_ties_small_k():
    _assert_parity(["identical"] * 100, 0.5, 3, "simple_ratio")


def test_fuzzy_topk_blockwise_parity_short_values():
    rng = np.random.default_rng(7)
    values = [f"s{rng.integers(0, 100):03d}" for _ in range(50)]

    _assert_parity(values, 0.9, 5, "simple_ratio")


def test_fuzzy_topk_blockwise_parity_single_value():
    _assert_parity(["s000"], 0.9, 5, "simple_ratio")


def test_fuzzy_topk_blockwise_parity_two_values():
    _assert_parity(["s000", "s001"], 0.9, 5, "simple_ratio")


def test_fuzzy_topk_blockwise_parity_fewer_values_than_a_block():
    """A strip wider than the state width degenerates without error."""
    rng = np.random.default_rng(7)
    values = [f"s{rng.integers(0, 100):03d}" for _ in range(7)]

    _assert_parity(values, 0.99, 2, "simple_ratio")


def test_fuzzy_topk_blockwise_parity_exactly_at_a_real_boundary():
    """The threshold sits exactly on a real pair score; the tolerant
    boundary comparison admits it identically on both paths."""
    values = []
    stems = ["kingstreet", "harborview", "millpond", "eastgate", "elmrow"]
    for c in range(30):
        for j in range(10):
            stem = stems[(c + j) % len(stems)]
            values.append(stem + ("x" * (j % 7)) + str(c * 31 + j))

    scorer = lk.fuzzy().get_scorer()
    scores = process.cdist([values[0]], values, scorer=scorer, dtype="float64")
    boundary = float(np.max(np.asarray(scores)[np.asarray(scores) < 100])) / 100

    _assert_parity(values, boundary, 5, "simple_ratio")
    _assert_parity(values, boundary - 1e-10, 5, "simple_ratio")


def test_fuzzy_topk_blockwise_parity_50k_scale():
    """Scale parity on the benchmark-shaped workload (slow, ~80s)."""
    rng = np.random.default_rng(1234)
    from string import ascii_lowercase

    words = list(ascii_lowercase[:8])
    values = [f"{'-'.join(rng.choice(words, size=3))}-{rng.integers(0, 10**6):06d}" for _ in range(50_000)]

    _assert_parity(values, 0.95, 5, "simple_ratio")


###############################
# mechanism: block machinery  #
##############################


def test_fuzzy_topk_uses_cdist_not_extract(monkeypatch):
    """The top-k path scores through `process.cdist` strips, not per-row
    `process.extract`."""
    from rapidfuzz import process

    calls = {"cdist": 0, "extract": 0}
    real_cdist, real_extract = process.cdist, process.extract

    def spy_cdist(*args, **kwargs):
        calls["cdist"] += 1
        return real_cdist(*args, **kwargs)

    def spy_extract(*args, **kwargs):
        calls["extract"] += 1
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(process, "cdist", spy_cdist)
    monkeypatch.setattr(process, "extract", spy_extract)

    array = pa.array(["abcd", "abce", "abcf", "zzzz"])
    list(lk.fuzzy(threshold=0.7, topn=2)._gen_similarity_pairs(array))

    assert calls["extract"] == 0
    assert calls["cdist"] >= 1


def test_fuzzy_topk_cdist_receives_resolved_workers(monkeypatch):
    """`workers` reaches the top-k block calls like the all-pairs path."""
    from rapidfuzz import process

    seen_workers = []
    real_cdist = process.cdist

    def spy_cdist(*args, **kwargs):
        seen_workers.append(kwargs.get("workers"))
        return real_cdist(*args, **kwargs)

    monkeypatch.setattr(process, "cdist", spy_cdist)

    array = pa.array(["abcd", "abce", "abcf", "zzzz"])
    list(lk.fuzzy(threshold=0.7, topn=2, workers=2)._gen_similarity_pairs(array))

    assert seen_workers and all(w == 2 for w in seen_workers)


def test_fuzzy_topk_cdist_runs_in_float64(monkeypatch):
    """The block calls score in float64: float32 rounding reorders ties.

    Precedent: the edit-distance block path (50ceb98).
    """
    from rapidfuzz import process

    seen_dtypes = []
    real_cdist = process.cdist

    def spy_cdist(*args, **kwargs):
        seen_dtypes.append(kwargs.get("dtype"))
        return real_cdist(*args, **kwargs)

    monkeypatch.setattr(process, "cdist", spy_cdist)

    array = pa.array(["abcd", "abce", "abcf", "zzzz"])
    list(lk.fuzzy(threshold=0.7, topn=2)._gen_similarity_pairs(array))

    assert seen_dtypes and all(d == "float64" for d in seen_dtypes)


def test_fuzzy_topk_max_block_memory_bounds_strips():
    """A tiny `max_block_memory` still produces identical pairs."""
    array = pa.array(["abcd", "abce", "abcf", "zzzz", "abcm", "abcn"])

    capped = list(lk.fuzzy(threshold=0.7, topn=2, max_block_memory=256)._gen_similarity_pairs(array))
    reference = _reference_topk_pairs(
        ["abcd", "abce", "abcf", "zzzz", "abcm", "abcn"],
        list(range(6)),
        0.7,
        lk.fuzzy().get_scorer(),
        2,
    )

    assert capped == reference
