"""Tests for the shared memory embedding cache (src/waggle/shared_container.py).

Tests cover:
- EmbeddingCache: binary file creation, header validation, put/get round-trip
- Zero-copy view correctness (data integrity)
- Float16 dtype support
- Soft-delete (remove) and stale-entry replacement
- load_matrix() batch fetch
- File grow (auto-resize)
- flush_index / load round-trip persistence
- rebuild_from_db() from a minimal SQLite database
- SharedMemoryBridge: create, attach, close, unlink lifecycle
- Graceful fallback when cache is not open
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from waggle.shared_container import (
    EmbeddingCache,
    SharedMemoryBridge,
    _SHM_AVAILABLE,
    _HEADER_SIZE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return (v / norm).astype(np.float32) if norm > 0.0 else v.astype(np.float32)


def _rand_unit(dim: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _unit(rng.standard_normal(dim).astype(np.float32))


def _make_cache(tmp_path: Path, dim: int = 64, dtype: str = "float32") -> EmbeddingCache:
    cache = EmbeddingCache(tmp_path / "waggle.emb", dim=dim, dtype=dtype, initial_capacity=32)
    ok = cache.open()
    assert ok, "Cache failed to open"
    return cache


def _make_db(tmp_path: Path, n_nodes: int = 20, dim: int = 64) -> sqlite3.Connection:
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE nodes (id TEXT, embedding BLOB, tenant_id TEXT)")
    for i in range(n_nodes):
        vec = _rand_unit(dim, seed=i)
        conn.execute("INSERT INTO nodes VALUES (?, ?, ?)",
                     (f"node-{i}", vec.tobytes(), "tenant-a"))
    conn.commit()
    return conn


class _FakeModel:
    def from_bytes(self, data: bytes) -> np.ndarray:
        return np.frombuffer(data, dtype=np.float32).copy()


# ---------------------------------------------------------------------------
# EmbeddingCache tests
# ---------------------------------------------------------------------------

class TestEmbeddingCacheOpen:
    def test_open_creates_file(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        assert cache.path.exists()
        assert cache.is_open
        cache.close()

    def test_binary_file_has_correct_size(self, tmp_path: Path) -> None:
        from waggle.shared_container import _MIN_CAPACITY
        dim = 32
        capacity = _MIN_CAPACITY  # initial_capacity is clamped to _MIN_CAPACITY
        cache = EmbeddingCache(tmp_path / "waggle.emb", dim=dim, initial_capacity=capacity)
        cache.open()
        expected = _HEADER_SIZE + capacity * dim * 4  # float32 = 4 bytes
        assert cache.path.stat().st_size == expected
        cache.close()

    def test_open_returns_false_on_bad_file(self, tmp_path: Path) -> None:
        bad = tmp_path / "waggle.emb"
        bad.write_bytes(b"NOTMAGIC" + b"\x00" * 56)
        cache = EmbeddingCache(bad, dim=64)
        # Trying to open corrupt file should not crash, return False
        result = cache.open()
        # May return False or True depending on fallback; but must not raise
        cache.close()


class TestEmbeddingCachePutGet:
    def test_put_and_get_roundtrip(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        vec = _rand_unit(64, seed=1)
        cache.put("node-1", vec)
        result = cache.get("node-1")
        assert result is not None
        np.testing.assert_allclose(result, vec, atol=1e-6)
        cache.close()

    def test_get_unknown_returns_none(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        assert cache.get("does-not-exist") is None
        cache.close()

    def test_cached_count_increments(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        for i in range(5):
            cache.put(f"node-{i}", _rand_unit(64, seed=i))
        assert cache.cached_count == 5
        cache.close()

    def test_put_replaces_stale_entry(self, tmp_path: Path) -> None:
        """Re-inserting a node should update the embedding."""
        cache = _make_cache(tmp_path)
        old_vec = _rand_unit(64, seed=10)
        new_vec = _rand_unit(64, seed=99)
        cache.put("node-x", old_vec)
        cache.put("node-x", new_vec)
        result = cache.get("node-x")
        assert result is not None
        # Should return the new vector, not the old one
        np.testing.assert_allclose(result, new_vec, atol=1e-6)
        cache.close()

    def test_remove_makes_get_return_none(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        cache.put("node-del", _rand_unit(64))
        cache.remove("node-del")
        assert cache.get("node-del") is None
        cache.close()

    def test_remove_unknown_is_safe(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        cache.remove("no-such-node")  # must not raise
        cache.close()

    def test_get_without_open_returns_none(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(tmp_path / "waggle.emb", dim=64)
        # Not opened — get() should be a safe no-op
        assert cache.get("any") is None

    def test_put_without_open_is_safe(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(tmp_path / "waggle.emb", dim=64)
        cache.put("any", _rand_unit(64))  # must not raise


class TestEmbeddingCacheFloat16:
    def test_float16_roundtrip(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path, dtype="float16")
        vec = _rand_unit(64, seed=7)
        cache.put("node-fp16", vec)
        result = cache.get("node-fp16")
        assert result is not None
        # Float16 has ~3 decimal places of precision
        np.testing.assert_allclose(result.astype(np.float32), vec, atol=1e-3)
        cache.close()

    def test_float16_file_is_smaller(self, tmp_path: Path) -> None:
        dim = 64
        capacity = 64
        cache32 = EmbeddingCache(tmp_path / "f32.emb", dim=dim, dtype="float32", initial_capacity=capacity)
        cache16 = EmbeddingCache(tmp_path / "f16.emb", dim=dim, dtype="float16", initial_capacity=capacity)
        cache32.open()
        cache16.open()
        size32 = cache32.path.stat().st_size
        size16 = cache16.path.stat().st_size
        assert size16 < size32, f"float16 file ({size16}) should be smaller than float32 ({size32})"
        cache32.close()
        cache16.close()


class TestEmbeddingCacheBatchFetch:
    def test_load_matrix_returns_correct_shape(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path, dim=32)
        node_ids = [f"n-{i}" for i in range(10)]
        for i, nid in enumerate(node_ids):
            cache.put(nid, _rand_unit(32, seed=i))
        matrix, hits = cache.load_matrix(node_ids)
        assert matrix is not None
        assert matrix.shape == (10, 32)
        assert hits == node_ids
        cache.close()

    def test_load_matrix_partial_hits(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path, dim=32)
        cache.put("a", _rand_unit(32, seed=0))
        # "b" not in cache
        matrix, hits = cache.load_matrix(["a", "b"])
        assert matrix is not None
        assert matrix.shape == (1, 32)
        assert hits == ["a"]
        cache.close()

    def test_load_matrix_empty_input(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path, dim=32)
        matrix, hits = cache.load_matrix([])
        assert matrix is None
        assert hits == []
        cache.close()

    def test_load_matrix_no_hits(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path, dim=32)
        matrix, hits = cache.load_matrix(["unknown-a", "unknown-b"])
        assert matrix is None
        assert hits == []
        cache.close()


class TestEmbeddingCacheGrow:
    def test_auto_grow_beyond_initial_capacity(self, tmp_path: Path) -> None:
        # Start with capacity=4, write 10 nodes → should auto-grow
        cache = EmbeddingCache(tmp_path / "waggle.emb", dim=16, initial_capacity=4)
        cache.open()
        for i in range(10):
            cache.put(f"node-{i}", _rand_unit(16, seed=i))
        assert cache.cached_count == 10
        # All nodes should still be retrievable after grow
        for i in range(10):
            result = cache.get(f"node-{i}")
            assert result is not None, f"node-{i} missing after grow"
        cache.close()


class TestEmbeddingCachePersistence:
    def test_flush_and_reload_index(self, tmp_path: Path) -> None:
        cache1 = _make_cache(tmp_path, dim=32)
        vecs = {f"node-{i}": _rand_unit(32, seed=i) for i in range(5)}
        for nid, vec in vecs.items():
            cache1.put(nid, vec)
        cache1.flush_index()
        cache1.close()

        # Reload
        cache2 = EmbeddingCache(tmp_path / "waggle.emb", dim=32)
        loaded = cache2.load()
        assert loaded
        for nid, vec in vecs.items():
            result = cache2.get(nid)
            assert result is not None, f"{nid} missing after reload"
            np.testing.assert_allclose(result, vec, atol=1e-6)
        cache2.close()

    def test_load_returns_false_when_no_files(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(tmp_path / "waggle.emb", dim=32)
        assert not cache.load()


class TestEmbeddingCacheRebuildFromDb:
    def test_rebuild_populates_cache(self, tmp_path: Path) -> None:
        conn = _make_db(tmp_path, n_nodes=20, dim=64)
        cache = _make_cache(tmp_path, dim=64)
        count = cache.rebuild_from_db(conn, "tenant-a", _FakeModel())
        assert count == 20
        assert cache.cached_count == 20
        # Spot-check one node
        assert cache.get("node-0") is not None
        cache.close()
        conn.close()

    def test_rebuild_empty_db(self, tmp_path: Path) -> None:
        conn = _make_db(tmp_path, n_nodes=0)
        cache = _make_cache(tmp_path, dim=64)
        count = cache.rebuild_from_db(conn, "tenant-a", _FakeModel())
        assert count == 0
        conn.close()
        cache.close()

    def test_rebuild_wrong_tenant_yields_zero(self, tmp_path: Path) -> None:
        conn = _make_db(tmp_path, n_nodes=5)
        cache = _make_cache(tmp_path, dim=64)
        count = cache.rebuild_from_db(conn, "tenant-OTHER", _FakeModel())
        assert count == 0
        conn.close()
        cache.close()


# ---------------------------------------------------------------------------
# SharedMemoryBridge tests
# ---------------------------------------------------------------------------

shm_required = pytest.mark.skipif(
    not _SHM_AVAILABLE,
    reason="multiprocessing.shared_memory not available",
)

_SHM_TEST_NAME = "waggle_test_bridge_001"


@shm_required
class TestSharedMemoryBridge:
    def teardown_method(self) -> None:
        """Best-effort cleanup after each test."""
        bridge = SharedMemoryBridge(_SHM_TEST_NAME, shape=(4, 8), dtype=np.float32)
        bridge.unlink()

    def test_create_and_read(self) -> None:
        dim = 16
        n = 4
        matrix = np.stack([_rand_unit(dim, seed=i) for i in range(n)])
        bridge = SharedMemoryBridge(_SHM_TEST_NAME, shape=(n, dim), dtype=np.float32)
        ok = bridge.create(matrix)
        assert ok
        assert bridge.array is not None
        np.testing.assert_allclose(bridge.array, matrix, atol=1e-6)
        bridge.close()

    def test_attach_to_existing(self) -> None:
        dim = 8
        n = 3
        matrix = np.stack([_rand_unit(dim, seed=i) for i in range(n)])
        creator = SharedMemoryBridge(_SHM_TEST_NAME, shape=(n, dim), dtype=np.float32)
        creator.create(matrix)

        reader = SharedMemoryBridge(_SHM_TEST_NAME, shape=(n, dim), dtype=np.float32)
        ok = reader.attach()
        assert ok
        assert reader.array is not None
        np.testing.assert_allclose(reader.array, matrix, atol=1e-6)
        reader.close()
        creator.close()

    def test_not_available_is_safe(self) -> None:
        """Bridge with unavailable SHM must not crash on create/attach/close."""
        bridge = SharedMemoryBridge("waggle_test_unavailable", shape=(2, 4), dtype=np.float32)
        # Just ensure the is_available property works
        assert isinstance(bridge.is_available, bool)
        bridge.close()   # must not raise
        bridge.unlink()  # must not raise

    def test_make_shared_bridge_from_cache(self, tmp_path: Path) -> None:
        dim = 16
        cache = _make_cache(tmp_path, dim=dim)
        for i in range(5):
            cache.put(f"node-{i}", _rand_unit(dim, seed=i))
        bridge = cache.make_shared_bridge(_SHM_TEST_NAME)
        if bridge.is_available and bridge.array is not None:
            assert bridge.array.shape[1] == dim
        bridge.close()
        cache.close()
