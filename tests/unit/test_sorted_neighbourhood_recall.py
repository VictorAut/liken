"""Recall characterization for the `sorted_neighbourhood` deduper.

Unlike the `knn` floor suite, there is no hard recall floor here: recall
loss below `window = n` is inherent to the window shape (pairs outside
each other's windows are never scored), not a broken mechanism. The tests
assert only structural sanity — the full-window degeneracy and recall
monotonicity — and the measured deviation numbers are recorded in the
work item's `verification.md`.
"""

import random
import string
from itertools import pairwise

import pyarrow as pa
import pytest

import liken as lk


THRESHOLD = 0.8


def _graded_clusters(n_clusters: int, members_per_cluster: int, *, seed: int) -> list[str]:
    """Clustered strings with graded typo drift.

    Each cluster holds a base token plus `members_per_cluster` members;
    member j's value has j single-character substitutions in later
    positions, so within-cluster similarity falls strictly with j and
    clusters are not all-ties. Bases are spread across the alphabet so
    different clusters sort far apart.
    """
    rng = random.Random(seed)
    rows: list[str] = []
    for _ in range(n_clusters):
        base = "".join(rng.choice(string.ascii_lowercase) for _ in range(12))
        rows.append(base)
        for step in range(1, members_per_cluster + 1):
            chars = list(base)
            for _ in range(step):
                pos = rng.randrange(6, len(chars))  # later positions only
                chars[pos] = rng.choice([c for c in string.ascii_lowercase if c != chars[pos]])
            rows.append("".join(chars))
    rng.shuffle(rows)
    return rows


def test_sorted_neighbourhood_full_window_equals_all_pairs_fuzzy():
    """`window >= n` degenerates to all-pairs: the pair set matches `fuzzy` exactly.

    The D10 anchor — with the window no longer restricting generation, the
    deduper must reproduce `fuzzy` pair-for-pair, including the missing
    star pairs.
    """
    rows = [*_graded_clusters(8, 4, seed=51), None, None]
    array = pa.array(rows)
    n = len(rows)

    fuzzy_pairs = set(lk.fuzzy(threshold=THRESHOLD)._gen_similarity_pairs(array))
    sn_pairs = set(lk.sorted_neighbourhood(threshold=THRESHOLD, window=n)._gen_similarity_pairs(array))

    assert fuzzy_pairs
    assert sn_pairs == fuzzy_pairs


def test_sorted_neighbourhood_recall_non_decreasing_in_window():
    """Recall vs all-pairs `fuzzy` never decreases as the window widens.

    A wider window is a strict superset of scored pairs, so recovery can
    only grow. A drop between window sizes indicates a broken window
    scan (duplicated or skipped pairs).
    """
    rows = _graded_clusters(12, 6, seed=77)
    array = pa.array(rows)
    n = len(rows)

    reference = set(lk.fuzzy(threshold=THRESHOLD)._gen_similarity_pairs(array))
    assert reference  # the fixture produces pairs at all

    recalls = []
    for window in (2, 3, 5, 8, n):
        recovered = set(lk.sorted_neighbourhood(threshold=THRESHOLD, window=window)._gen_similarity_pairs(array))
        recalls.append(len(recovered & reference) / len(reference))

    assert all(b >= a for a, b in pairwise(recalls))


@pytest.mark.parametrize("n_clusters", [5, 10, 20], ids=["5c", "10c", "20c"])
def test_sorted_neighbourhood_characterization_cluster_spread(n_clusters):
    """Clustered workload at fixed window: recovery is recorded, not asserted.

    Clusters of 12 members straddle a window of 5, the regime the
    docstring documents: pairs between members further than the window
    apart in sorted order are never scored. Recovery is a property of
    window width and cluster sort spread, not of the code; the measured
    numbers are recorded in the work item's verification.md (shared
    measurement obligation, cross-referenced with WI049/WI050).
    """
    rows = _graded_clusters(n_clusters, 11, seed=99)
    array = pa.array(rows)

    reference = set(lk.fuzzy(threshold=THRESHOLD)._gen_similarity_pairs(array))
    recovered = set(lk.sorted_neighbourhood(threshold=THRESHOLD, window=5)._gen_similarity_pairs(array))

    assert reference  # the fixture produces pairs at all
    recall = len(recovered & reference) / len(reference)
    # sanity bound only: a working window scan recovers something
    assert recall > 0.0
