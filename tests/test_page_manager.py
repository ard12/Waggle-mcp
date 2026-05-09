"""Tests for the spatial graph paging module (src/waggle/page_manager.py).

Tests cover:
- K-means clustering correctness (_kmeans_lloyd)
- GraphPageManager: build_pages, score_pages, get_page_boost, get_co_paged_nodes
- Incremental assign_new_node + remove_node
- save() / load() round-trip persistence
- rebuild_from_db() from a minimal SQLite database
- rebuild_needed() trigger conditions
- Graceful handling of empty / single-node inputs
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from waggle.page_manager import (
    PAGE_BOOST_CAP,
    GraphPageManager,
    _kmeans_lloyd,
    _safe_mean,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0.0 else v


def _rand_unit(dim: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _unit(rng.standard_normal(dim).astype(np.float32))


def _cluster_embeddings(
    n_clusters: int = 3,
    nodes_per_cluster: int = 20,
    dim: int = 32,
    noise: float = 0.05,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict[str, str], list[list[str]]]:
    """Generate clearly separable clustered node embeddings.

    Returns:
        embeddings:  {node_id: unit_vector}
        projects:    {node_id: project_name}  (all same project for simplicity)
        true_groups: [[node_ids in cluster 0], [cluster 1], ...]
    """
    rng = np.random.default_rng(seed)
    # Orthogonal base directions — maximally separated in a low-dim space
    bases = [_unit(rng.standard_normal(dim).astype(np.float32)) for _ in range(n_clusters)]
    embeddings: dict[str, np.ndarray] = {}
    projects: dict[str, str] = {}
    true_groups: list[list[str]] = [[] for _ in range(n_clusters)]

    for cluster_idx, base in enumerate(bases):
        for i in range(nodes_per_cluster):
            nid = f"c{cluster_idx}-n{i}"
            noise_vec = rng.standard_normal(dim).astype(np.float32) * noise
            vec = _unit(base + noise_vec)
            embeddings[nid] = vec
            projects[nid] = "proj-a"
            true_groups[cluster_idx].append(nid)

    return embeddings, projects, true_groups


def _make_pm(tmp_path: Path, **kwargs) -> GraphPageManager:
    return GraphPageManager(tmp_path / "waggle.pages.json", **kwargs)


# ---------------------------------------------------------------------------
# _kmeans_lloyd unit tests
# ---------------------------------------------------------------------------

class TestKMeansLloyd:
    def test_basic_clustering(self) -> None:
        """Clearly separated clusters should be assigned correctly."""
        rng = np.random.default_rng(0)
        dim = 16
        # Two clusters far apart
        c0 = _unit(np.array([1.0] + [0.0] * (dim - 1), dtype=np.float32))
        c1 = _unit(np.array([0.0, 1.0] + [0.0] * (dim - 2), dtype=np.float32))
        pts_a = np.stack([_unit(c0 + rng.standard_normal(dim).astype(np.float32) * 0.01) for _ in range(10)])
        pts_b = np.stack([_unit(c1 + rng.standard_normal(dim).astype(np.float32) * 0.01) for _ in range(10)])
        matrix = np.vstack([pts_a, pts_b])
        labels, centroids = _kmeans_lloyd(matrix, n_clusters=2)
        assert labels.shape == (20,)
        # All points in each half should share the same label
        assert len(set(labels[:10])) == 1
        assert len(set(labels[10:])) == 1
        assert labels[0] != labels[10]  # different clusters
        assert centroids.shape == (2, dim)

    def test_returns_normalised_centroids(self) -> None:
        matrix = np.stack([_rand_unit(32, seed=i) for i in range(20)])
        _, centroids = _kmeans_lloyd(matrix, n_clusters=3)
        norms = np.linalg.norm(centroids, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_n_clusters_gt_n_clamped(self) -> None:
        """Requesting more clusters than points should not crash."""
        matrix = np.stack([_rand_unit(16, seed=i) for i in range(5)])
        labels, centroids = _kmeans_lloyd(matrix, n_clusters=100)
        assert labels.shape == (5,)
        assert centroids.shape[0] <= 5

    def test_single_cluster(self) -> None:
        matrix = np.stack([_rand_unit(16, seed=i) for i in range(10)])
        labels, centroids = _kmeans_lloyd(matrix, n_clusters=1)
        assert set(labels) == {0}
        assert centroids.shape == (1, 16)

    def test_reproducible_with_same_seed(self) -> None:
        matrix = np.stack([_rand_unit(32, seed=i) for i in range(30)])
        labels1, _ = _kmeans_lloyd(matrix, n_clusters=3, seed=7)
        labels2, _ = _kmeans_lloyd(matrix, n_clusters=3, seed=7)
        np.testing.assert_array_equal(labels1, labels2)


# ---------------------------------------------------------------------------
# GraphPageManager unit tests
# ---------------------------------------------------------------------------

class TestGraphPageManager:
    def test_not_built_before_build(self, tmp_path: Path) -> None:
        pm = _make_pm(tmp_path)
        assert not pm.is_built
        assert pm.page_count == 0

    def test_build_pages_basic(self, tmp_path: Path) -> None:
        embeddings, projects, _ = _cluster_embeddings(n_clusters=3, nodes_per_cluster=20)
        pm = _make_pm(tmp_path, page_size=20)
        pm.build_pages(embeddings, projects)
        assert pm.is_built
        assert pm.page_count >= 1
        assert pm.node_count == 60

    def test_all_nodes_assigned(self, tmp_path: Path) -> None:
        embeddings, projects, _ = _cluster_embeddings(n_clusters=2, nodes_per_cluster=15)
        pm = _make_pm(tmp_path, page_size=15)
        pm.build_pages(embeddings, projects)
        for nid in embeddings:
            assert nid in pm._node_page, f"{nid} was not assigned to any page"

    def test_project_partitioned(self, tmp_path: Path) -> None:
        """Nodes from different projects must never end up in the same page."""
        embeddings: dict[str, np.ndarray] = {}
        projects: dict[str, str] = {}
        for i in range(20):
            embeddings[f"a-{i}"] = _rand_unit(32, seed=i)
            projects[f"a-{i}"] = "proj-alpha"
        for i in range(20):
            embeddings[f"b-{i}"] = _rand_unit(32, seed=100 + i)
            projects[f"b-{i}"] = "proj-beta"

        pm = _make_pm(tmp_path, page_size=20)
        pm.build_pages(embeddings, projects)

        # Collect pages containing proj-alpha and proj-beta nodes
        alpha_pages = {pm._node_page[f"a-{i}"] for i in range(20)}
        beta_pages = {pm._node_page[f"b-{i}"] for i in range(20)}
        # The two sets must be disjoint (different project → different page)
        assert alpha_pages.isdisjoint(beta_pages), "Cross-project page assignment detected"

    def test_score_pages_top_hit(self, tmp_path: Path) -> None:
        """Querying with a cluster centroid should give that cluster's page the top score."""
        embeddings, projects, true_groups = _cluster_embeddings(
            n_clusters=3, nodes_per_cluster=20, dim=32
        )
        pm = _make_pm(tmp_path, page_size=20)
        pm.build_pages(embeddings, projects)

        # Use one of the nodes in group 0 as the query — its page should score highest.
        query_vec = embeddings[true_groups[0][0]]
        page_scores = pm.score_pages(query_vec)
        assert page_scores, "page_scores must not be empty after build"

        top_page = max(page_scores, key=lambda pid: page_scores[pid])
        # The top page should contain the query node itself.
        assert true_groups[0][0] in pm._pages[top_page]

    def test_score_pages_empty_before_build(self, tmp_path: Path) -> None:
        pm = _make_pm(tmp_path)
        result = pm.score_pages(_rand_unit(32))
        assert result == {}

    def test_score_pages_in_range(self, tmp_path: Path) -> None:
        embeddings, projects, _ = _cluster_embeddings()
        pm = _make_pm(tmp_path, page_size=20)
        pm.build_pages(embeddings, projects)
        for v in pm.score_pages(_rand_unit(32)).values():
            assert 0.0 <= v <= 1.0

    def test_get_page_boost_zero_before_build(self, tmp_path: Path) -> None:
        pm = _make_pm(tmp_path)
        boost = pm.get_page_boost("any-node", {0: 0.9})
        assert boost == 0.0

    def test_get_page_boost_non_negative(self, tmp_path: Path) -> None:
        embeddings, projects, true_groups = _cluster_embeddings()
        pm = _make_pm(tmp_path, page_size=20)
        pm.build_pages(embeddings, projects)
        query_vec = embeddings[true_groups[0][0]]
        page_scores = pm.score_pages(query_vec)
        for nid in list(embeddings)[:5]:
            boost = pm.get_page_boost(nid, page_scores)
            assert 0.0 <= boost <= PAGE_BOOST_CAP

    def test_get_co_paged_nodes_returns_neighbours(self, tmp_path: Path) -> None:
        embeddings, projects, _ = _cluster_embeddings(n_clusters=2, nodes_per_cluster=10)
        pm = _make_pm(tmp_path, page_size=10)
        pm.build_pages(embeddings, projects)

        # Pick one node and find its co-paged neighbours
        seed_node = "c0-n0"
        page_id = pm._node_page[seed_node]
        expected_neighbours = set(pm._pages[page_id]) - {seed_node}

        co_paged = pm.get_co_paged_nodes([seed_node])
        assert co_paged == expected_neighbours

    def test_get_co_paged_nodes_empty_input(self, tmp_path: Path) -> None:
        embeddings, projects, _ = _cluster_embeddings()
        pm = _make_pm(tmp_path, page_size=20)
        pm.build_pages(embeddings, projects)
        assert pm.get_co_paged_nodes([]) == set()

    def test_assign_new_node(self, tmp_path: Path) -> None:
        embeddings, projects, true_groups = _cluster_embeddings(
            n_clusters=2, nodes_per_cluster=10
        )
        pm = _make_pm(tmp_path, page_size=10)
        pm.build_pages(embeddings, projects)

        # New node very close to cluster 0 centroid
        new_vec = embeddings[true_groups[0][0]]  # same as existing cluster 0 node
        pm.assign_new_node("new-node", new_vec, "proj-a")

        assert "new-node" in pm._node_page
        # Should have landed in cluster 0's page, not cluster 1's
        assigned_page = pm._node_page["new-node"]
        cluster0_pages = {pm._node_page[nid] for nid in true_groups[0]}
        assert assigned_page in cluster0_pages

    def test_assign_new_node_no_pages_is_safe(self, tmp_path: Path) -> None:
        pm = _make_pm(tmp_path)
        result = pm.assign_new_node("orphan", _rand_unit(32), "proj")
        assert result is None
        assert pm._orphan_count == 1

    def test_remove_node(self, tmp_path: Path) -> None:
        embeddings, projects, _ = _cluster_embeddings(n_clusters=1, nodes_per_cluster=10)
        pm = _make_pm(tmp_path, page_size=10)
        pm.build_pages(embeddings, projects)

        target = "c0-n0"
        pm.remove_node(target)
        assert target not in pm._node_page
        for page in pm._pages.values():
            assert target not in page

    def test_remove_unknown_node_is_safe(self, tmp_path: Path) -> None:
        pm = _make_pm(tmp_path)
        pm.remove_node("does-not-exist")  # must not raise

    def test_rebuild_needed_not_built(self, tmp_path: Path) -> None:
        pm = _make_pm(tmp_path)
        assert pm.rebuild_needed(100)

    def test_rebuild_needed_false_when_fresh(self, tmp_path: Path) -> None:
        embeddings, projects, _ = _cluster_embeddings(n_clusters=2, nodes_per_cluster=20)
        pm = _make_pm(tmp_path, page_size=20, rebuild_threshold=1000)
        pm.build_pages(embeddings, projects)
        assert not pm.rebuild_needed(pm.node_count)

    def test_rebuild_needed_growth_trigger(self, tmp_path: Path) -> None:
        embeddings, projects, _ = _cluster_embeddings(n_clusters=2, nodes_per_cluster=20)
        pm = _make_pm(tmp_path, page_size=20, rebuild_threshold=1000)
        pm.build_pages(embeddings, projects)
        # Simulate 2× growth — should trigger rebuild
        assert pm.rebuild_needed(pm.node_count * 3)


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_creates_file(self, tmp_path: Path) -> None:
        embeddings, projects, _ = _cluster_embeddings()
        pm = _make_pm(tmp_path, page_size=20)
        pm.build_pages(embeddings, projects)
        pm.save()
        assert (tmp_path / "waggle.pages.json").exists()

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        embeddings, projects, true_groups = _cluster_embeddings(
            n_clusters=3, nodes_per_cluster=20
        )
        pm1 = _make_pm(tmp_path, page_size=20)
        pm1.build_pages(embeddings, projects)
        pm1.save()

        pm2 = _make_pm(tmp_path, page_size=20)
        loaded = pm2.load()

        assert loaded
        assert pm2.page_count == pm1.page_count
        assert pm2.node_count == pm1.node_count
        # Node→page assignments must match
        for nid in embeddings:
            assert pm2._node_page[nid] == pm1._node_page[nid]

    def test_load_wrong_version_returns_false(self, tmp_path: Path) -> None:
        pages_path = tmp_path / "waggle.pages.json"
        pages_path.write_text(json.dumps({"version": 99, "pages": {}, "node_page": {},
                                           "centroid_page_ids": [], "centroids": None}))
        pm = _make_pm(tmp_path)
        assert not pm.load()

    def test_load_missing_file_returns_false(self, tmp_path: Path) -> None:
        pm = _make_pm(tmp_path)
        assert not pm.load()

    def test_score_pages_after_load(self, tmp_path: Path) -> None:
        embeddings, projects, true_groups = _cluster_embeddings(
            n_clusters=3, nodes_per_cluster=20
        )
        pm1 = _make_pm(tmp_path, page_size=20)
        pm1.build_pages(embeddings, projects)
        pm1.save()

        pm2 = _make_pm(tmp_path, page_size=20)
        pm2.load()

        query_vec = embeddings[true_groups[0][0]]
        scores1 = pm1.score_pages(query_vec)
        scores2 = pm2.score_pages(query_vec)

        assert scores1.keys() == scores2.keys()
        for pid in scores1:
            assert abs(scores1[pid] - scores2[pid]) < 1e-4


# ---------------------------------------------------------------------------
# rebuild_from_db tests
# ---------------------------------------------------------------------------

class TestRebuildFromDb:
    def _make_db(self, tmp_path: Path, n_nodes: int = 20) -> sqlite3.Connection:
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE nodes (id TEXT, project TEXT, embedding BLOB, tenant_id TEXT)"
        )
        for i in range(n_nodes):
            vec = _rand_unit(32, seed=i)
            conn.execute(
                "INSERT INTO nodes VALUES (?, ?, ?, ?)",
                (f"node-{i}", "proj-test", vec.tobytes(), "test-tenant"),
            )
        conn.commit()
        return conn

    class _FakeModel:
        def from_bytes(self, data: bytes) -> np.ndarray:
            return np.frombuffer(data, dtype=np.float32)

    def test_rebuild_populates_pages(self, tmp_path: Path) -> None:
        conn = self._make_db(tmp_path, n_nodes=20)
        pm = _make_pm(tmp_path, page_size=10)
        count = pm.rebuild_from_db(conn, "test-tenant", self._FakeModel())
        assert count == 20
        assert pm.is_built
        assert pm.node_count == 20
        conn.close()

    def test_rebuild_saves_to_disk(self, tmp_path: Path) -> None:
        conn = self._make_db(tmp_path, n_nodes=10)
        pm = _make_pm(tmp_path, page_size=10)
        pm.rebuild_from_db(conn, "test-tenant", self._FakeModel())
        assert (tmp_path / "waggle.pages.json").exists()
        conn.close()

    def test_rebuild_empty_db(self, tmp_path: Path) -> None:
        conn = self._make_db(tmp_path, n_nodes=0)
        pm = _make_pm(tmp_path, page_size=10)
        count = pm.rebuild_from_db(conn, "test-tenant", self._FakeModel())
        assert count == 0
        assert not pm.is_built
        conn.close()
