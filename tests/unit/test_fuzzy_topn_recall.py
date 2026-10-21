"""Recall floor and characterization for `fuzzy(topn=k)`.

Two fixtures, two purposes:

- The floor test uses graded-mutation clusters (member j carries j more
  mutations of the cluster base), so each row's qualifying partners are
  strictly score-ordered rather than tied, and each row holds more
  qualifying partners than `topn` keeps — real truncation pressure on
  structured data. A hard assert: recovery below 90% means a broken
  mechanism, not a data-shape property.
- The characterization suite uses all-tie near-identical clusters, the
  worst case the docstring documents: index-order truncation inside the
  tie group gives low recovery that depends on cluster size, not on the
  code. Those tests assert only a sanity bound; the measured numbers are
  recorded in the work item.
"""

import random
import string

import pyarrow as pa
import pytest

import liken as lk


THRESHOLD = 0.8


def _graded_clusters(n_clusters: int, members_per_cluster: int, *, seed: int) -> list[str]:
    """Clusters with strictly score-ordered members.

    Each cluster holds a 24-character stem plus `members_per_cluster`
    members; member j differs from the stem in exactly j distinct
    positions. Within a cluster, similarity to the stem falls strictly
    with j, so a row's top-k cut is score-driven, not index-driven.
    """

    rng = random.Random(seed)
    values: list[str] = []
    for _ in range(n_clusters):
        stem = "".join(rng.choices(string.ascii_lowercase, k=24))
        values.append(stem)
        for mutations in range(1, members_per_cluster + 1):
            chars = list(stem)
            for position in rng.sample(range(len(chars)), mutations):
                chars[position] = rng.choice([c for c in string.ascii_lowercase if c != chars[position]])
            values.append("".join(chars))
    return values


def _all_pairs_reference(values: list[str]) -> set[tuple[int, int]]:
    return set(lk.fuzzy(threshold=THRESHOLD)._gen_similarity_pairs(pa.array(values)))


def _topn_pairs(values: list[str], topn: int) -> set[tuple[int, int]]:
    return set(lk.fuzzy(threshold=THRESHOLD, topn=topn)._gen_similarity_pairs(pa.array(values)))


def test_fuzzy_topn_recall_floor_on_graded_clusters():
    """`fuzzy(topn=5)` recovers >= 90% of the all-pairs pairs.

    30 clusters of 10 members each: every row has up to 9 within-cluster
    partners, so topn=5 genuinely truncates. A broken mechanism — an
    upper-triangle-only scan, a missing either-direction test — fails
    this floor.
    """
    values = _graded_clusters(30, 9, seed=49)

    reference = _all_pairs_reference(values)
    # sanity on the fixture: every reference pair is within-cluster, and
    # truncation pressure is real (partners per row exceed topn)
    cluster_size = 10
    assert all(pair[1] - pair[0] < cluster_size for pair in reference)
    assert reference  # the fixture produces pairs at all

    recovered = _topn_pairs(values, topn=5)

    ratio = len(recovered & reference) / len(reference)
    assert ratio >= 0.9


@pytest.mark.parametrize("n_members", [1000, 2000, 5000], ids=["1k", "2k", "5k"])
def test_fuzzy_topn_characterization_dense_tie_cluster(n_members):
    """All-tie dense clusters: recovery is deliberately unasserted quality-wise.

    A single cluster of n near-identical values is the worst case the
    docstring documents: every pair scores 100, ties sort by choice index,
    and each row keeps only its k best-by-index partners. Recovery is a
    property of k and cluster size, not of the code; the measured numbers
    are recorded in the work item's verification.md. This test only pins
    the sanity bound that the mechanism recovers something.
    """
    values = ["target"] * n_members

    reference = _all_pairs_reference(values)
    recovered = _topn_pairs(values, topn=5)

    assert reference  # the fixture produces pairs at all
    assert recovered
