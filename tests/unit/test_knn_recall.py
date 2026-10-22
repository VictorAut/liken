"""Recall floor and characterization for the `knn` deduper.

Two fixtures, two purposes, mirroring `test_fuzzy_topn_recall.py`:

- The floor test uses graded-drift clusters (member j's vectors sit
  strictly farther from the cluster centre), so each row's qualifying
  partners are strictly similarity-ordered rather than tied, and each row
  holds more qualifying partners than `k` keeps — real truncation
  pressure. A hard assert: recovery below 90% means a broken mechanism,
  not a data-shape property. The cKDTree pipeline is exact throughout,
  so a failure here is a code bug.
- The characterization suite uses all-tie near-identical clusters, the
  worst case the docstring documents: cKDTree tie order truncation gives
  low recovery that depends on cluster size, not on the code. Those
  tests assert only a sanity bound; the measured numbers are recorded in
  the work item.
"""

import numpy as np
import pyarrow as pa
import pytest

import liken as lk


THRESHOLD = 0.8


def _graded_clusters(n_clusters: int, members_per_cluster: int, *, seed: int) -> np.ndarray:
    """Clusters with strictly similarity-ordered members.

    Each cluster holds a unit-length centre plus `members_per_cluster`
    members; member j's vector is the centre displaced by j graded steps
    in its own random direction (a fan, not a line), so within-cluster
    similarity falls strictly with j and a row's top-k cut is
    score-driven, not index-driven. At these steps the widest within-
    cluster pairs still clear the threshold while cross-cluster pairs
    (up to cos ~ 0.7 on 32 dims) never do.
    """
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    for _ in range(n_clusters):
        centre = rng.normal(size=32)
        centre /= np.linalg.norm(centre)
        rows.append(centre)
        for step in range(1, members_per_cluster + 1):
            direction = rng.normal(size=32)
            direction /= np.linalg.norm(direction)
            rows.append(centre + direction * step * 0.12)
    return np.array(rows)


def _table(matrix: np.ndarray) -> pa.Table:
    return pa.table({f"col_{i}": matrix[:, i].tolist() for i in range(matrix.shape[1])})


def test_knn_recall_floor_on_graded_clusters():
    """`knn(k=5, metric='cosine')` recovers >= 90% of the all-pairs pairs.

    30 clusters of 10 members each: every row has up to 9 within-cluster
    partners, so k=5 genuinely truncates. A broken mechanism — an
    upper-triangle-only scan, a missing either-direction test, a wrong
    similarity conversion — fails this floor.
    """
    matrix = _graded_clusters(30, 9, seed=49)
    table = _table(matrix)

    reference = set(lk.cosine(threshold=THRESHOLD)._gen_similarity_pairs(table))
    # sanity on the fixture: every reference pair is within-cluster, and
    # truncation pressure is real (partners per row exceed k)
    cluster_size = 10
    assert all(pair[1] - pair[0] < cluster_size for pair in reference)
    assert reference  # the fixture produces pairs at all

    recovered = set(lk.knn(threshold=THRESHOLD, k=5, metric="cosine")._gen_similarity_pairs(table))

    ratio = len(recovered & reference) / len(reference)
    assert ratio >= 0.9


@pytest.mark.parametrize("n_members", [1000, 2000, 5000], ids=["1k", "2k", "5k"])
def test_knn_characterization_dense_tie_cluster(n_members):
    """All-tie dense clusters: recovery is deliberately unasserted quality-wise.

    A single cluster of n near-identical vectors is the worst case the
    docstring documents: every pair sits at distance ~0, ties resolve by
    cKDTree's own order, and each row keeps only its k best under that
    order. Recovery is a property of k and cluster size, not of the
    code; the measured numbers are recorded in the work item's
    verification.md. This test only pins the sanity bound that the
    mechanism recovers something.
    """
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(n_members, 3)) * 0.001  # one tight cluster
    table = _table(matrix)

    reference = set(lk.cosine(threshold=THRESHOLD)._gen_similarity_pairs(table))
    recovered = set(lk.knn(threshold=THRESHOLD, k=5, metric="cosine")._gen_similarity_pairs(table))

    assert reference  # the fixture produces pairs at all
    assert recovered
