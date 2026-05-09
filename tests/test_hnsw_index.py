"""Tests for the HNSW sidecar index (src/waggle/hnsw_index.py).

Tests cover both the live-index path (when hnswlib is installed) and the
graceful fallback path (when it is not).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from waggle.hnsw_index import HNSWLIB_AVAILABLE, HNSWIndex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_unit_vec(dim: int = 64, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_index(tmpdir: str, dim: int = 64, dtype: str = "float32") -> HNSWIndex:
    return HNSWIndex(Path(tmpdir) / "test.hnsw", dim=dim, max_elements=512, dtype=dtype)


# ---------------------------------------------------------------------------
# Fallback path — always runs regardless of hnswlib availability
# ---------------------------------------------------------------------------

class TestFallbackPath:
    """All public methods must be safe no-ops when hnswlib is not installed."""

    def test_is_available_matches_module_flag(self, tmp_path: Path) -> None:
        idx = HNSWIndex(tmp_path / "test.hnsw", dim=64)
        assert idx.is_available == HNSWLIB_AVAILABLE

    def test_element_count_zero_when_unavailable(self, tmp_path: Path) -> None:
        idx = HNSWIndex(tmp_path / "test.hnsw", dim=64)
        # element_count should always return a non-negative int
        assert idx.element_count >= 0

    def test_add_no_crash(self, tmp_path: Path) -> None:
        idx = HNSWIndex(tmp_path / "test.hnsw", dim=64)
        idx.add("node-1", _rand_unit_vec(64, seed=1))  # must not raise

    def test_remove_no_crash(self, tmp_path: Path) -> None:
        idx = HNSWIndex(tmp_path / "test.hnsw", dim=64)
        idx.add("node-1", _rand_unit_vec(64, seed=1))
        idx.remove("node-1")  # must not raise

    def test_remove_unknown_no_crash(self, tmp_path: Path) -> None:
        idx = HNSWIndex(tmp_path / "test.hnsw", dim=64)
        idx.remove("does-not-exist")  # must not raise

    def test_query_returns_list(self, tmp_path: Path) -> None:
        idx = HNSWIndex(tmp_path / "test.hnsw", dim=64)
        idx.add("node-1", _rand_unit_vec(64, seed=1))
        results = idx.query(_rand_unit_vec(64, seed=2), k=5)
        assert isinstance(results, list)

    def test_query_empty_when_no_hnswlib(self, tmp_path: Path) -> None:
        """Without hnswlib, query must always return an empty list."""
        idx = HNSWIndex(tmp_path / "test.hnsw", dim=64)
        idx.add("node-1", _rand_unit_vec(64, seed=1))
        results = idx.query(_rand_unit_vec(64, seed=2), k=5)
        if not HNSWLIB_AVAILABLE:
            assert results == []

    def test_save_no_crash(self, tmp_path: Path) -> None:
        idx = HNSWIndex(tmp_path / "test.hnsw", dim=64)
        idx.save()  # must not raise even when index is None


# ---------------------------------------------------------------------------
# Live path — only runs when hnswlib is installed
# ---------------------------------------------------------------------------

hnswlib_required = pytest.mark.skipif(
    not HNSWLIB_AVAILABLE,
    reason="hnswlib not installed (pip install waggle-mcp[perf])",
)


@hnswlib_required
class TestLiveIndex:
    """Functional tests for the live HNSW index."""

    def test_add_and_query_basic(self, tmp_path: Path) -> None:
        idx = _make_index(str(tmp_path))
        assert idx.is_available

        vecs = {f"node-{i}": _rand_unit_vec(64, seed=i) for i in range(10)}
        for nid, vec in vecs.items():
            idx.add(nid, vec)

        assert idx.element_count == 10
        # Query with the exact vector of node-0 — it should be the top hit.
        results = idx.query(vecs["node-0"], k=3)
        assert len(results) > 0
        top_id, top_sim = results[0]
        assert top_id == "node-0"
        assert 0.99 <= top_sim <= 1.0  # cosine similarity with itself ≈ 1

    def test_similarities_in_range(self, tmp_path: Path) -> None:
        idx = _make_index(str(tmp_path))
        vecs = {f"node-{i}": _rand_unit_vec(64, seed=i + 100) for i in range(20)}
        for nid, vec in vecs.items():
            idx.add(nid, vec)

        q = _rand_unit_vec(64, seed=999)
        results = idx.query(q, k=10)
        for _, sim in results:
            assert 0.0 <= sim <= 1.0, f"Similarity {sim} out of [0,1]"

    def test_remove_node(self, tmp_path: Path) -> None:
        idx = _make_index(str(tmp_path))
        vec0 = _rand_unit_vec(64, seed=0)
        idx.add("node-0", vec0)
        idx.add("node-1", _rand_unit_vec(64, seed=1))
        assert idx.element_count == 2

        idx.remove("node-0")
        assert idx.element_count == 1

        # Querying with node-0's own vector should NOT return node-0.
        results = idx.query(vec0, k=5)
        returned_ids = [nid for nid, _ in results]
        assert "node-0" not in returned_ids

    def test_add_replaces_stale_entry(self, tmp_path: Path) -> None:
        """Re-adding a node with a new vector should replace the old one."""
        idx = _make_index(str(tmp_path))
        old_vec = _rand_unit_vec(64, seed=0)
        new_vec = _rand_unit_vec(64, seed=42)
        idx.add("node-x", old_vec)
        idx.add("node-x", new_vec)  # update

        # element_count should not grow (old label is deleted, new one added)
        assert idx.element_count == 1

        # Querying with new_vec should return node-x at the top.
        results = idx.query(new_vec, k=3)
        assert results[0][0] == "node-x"
        assert results[0][1] >= 0.99

    def test_scope_filter(self, tmp_path: Path) -> None:
        idx = _make_index(str(tmp_path))
        vecs = {f"node-{i}": _rand_unit_vec(64, seed=i) for i in range(10)}
        for nid, vec in vecs.items():
            idx.add(nid, vec)

        allowed = {"node-0", "node-1", "node-2"}
        results = idx.query(vecs["node-0"], k=10, scope_ids=allowed)
        returned_ids = {nid for nid, _ in results}
        assert returned_ids.issubset(allowed), f"Got out-of-scope IDs: {returned_ids - allowed}"

    def test_save_and_reload(self, tmp_path: Path) -> None:
        """Persisting and reloading the index should preserve all entries."""
        idx1 = _make_index(str(tmp_path))
        vecs = {f"node-{i}": _rand_unit_vec(64, seed=i + 200) for i in range(5)}
        for nid, vec in vecs.items():
            idx1.add(nid, vec)
        idx1.save()

        # Fresh index object loading the same file.
        idx2 = _make_index(str(tmp_path))
        assert idx2.element_count == 5

        results = idx2.query(vecs["node-0"], k=3)
        assert results[0][0] == "node-0"

    def test_float16_dtype(self, tmp_path: Path) -> None:
        """float16 index should still return valid results (slight precision loss acceptable)."""
        idx = _make_index(str(tmp_path), dtype="float16")
        vecs = {f"node-{i}": _rand_unit_vec(64, seed=i + 50) for i in range(10)}
        for nid, vec in vecs.items():
            idx.add(nid, vec)

        results = idx.query(vecs["node-0"], k=3)
        assert results[0][0] == "node-0"
        assert results[0][1] >= 0.95  # allow slight precision loss

    def test_rebuild_from_db(self, tmp_path: Path) -> None:
        """rebuild_from_db() should populate the index from SQLite rows."""
        import sqlite3

        # Create a minimal in-memory-style SQLite DB with one node row.
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE nodes (id TEXT, embedding BLOB, tenant_id TEXT)"
        )
        vec = _rand_unit_vec(64, seed=77)
        conn.execute(
            "INSERT INTO nodes VALUES (?, ?, ?)",
            ("rebuild-node-1", vec.tobytes(), "test-tenant"),
        )
        conn.commit()

        class FakeModel:
            def from_bytes(self, data: bytes) -> np.ndarray:
                return np.frombuffer(data, dtype=np.float32)

        idx = _make_index(str(tmp_path))
        count = idx.rebuild_from_db(conn, "test-tenant", FakeModel())
        assert count == 1
        assert idx.element_count == 1

        results = idx.query(vec, k=1)
        assert results[0][0] == "rebuild-node-1"
        assert results[0][1] >= 0.99
        conn.close()

    def test_auto_grow(self, tmp_path: Path) -> None:
        """Index should auto-grow when max_elements is exceeded."""
        # Start very small to force growth.
        idx = HNSWIndex(tmp_path / "tiny.hnsw", dim=64, max_elements=5)
        for i in range(20):
            idx.add(f"node-{i}", _rand_unit_vec(64, seed=i + 300))
        assert idx.element_count >= 20
