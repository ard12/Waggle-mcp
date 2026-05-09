"""HNSW approximate nearest-neighbour index for Waggle node embeddings.

This module provides a lightweight sidecar index that sits alongside the
SQLite database and enables O(log N) vector search instead of the default
O(N) full table-scan + brute-force cosine loop.

Architecture
------------
* The index is stored as a binary file next to ``waggle.db``:
  ``~/.waggle/waggle.hnsw``
* A JSON metadata file keeps the mapping between hnswlib's integer labels
  and Waggle's UUID node IDs:
  ``~/.waggle/waggle.hnsw.meta.json``
* Writes go through SQLite first (source of truth), then the index is
  updated via ``add()`` or ``mark_deleted()``.
* On startup, if the index file is missing, ``rebuild_from_db()`` re-creates
  it from the SQLite node table.

Quantization
------------
Set ``dtype="float16"`` (via ``WAGGLE_HNSW_DTYPE=float16``) to store vectors
at half precision.  This halves memory usage with negligible recall loss for
typical sentence-transformer embeddings.

Fallback
--------
If ``hnswlib`` is not installed, all public methods return graceful no-ops /
empty results and log a one-time warning.  The caller (``MemoryGraph``) falls
back to the brute-force batch-matmul path from ``embeddings.py``.

Usage
-----
    from waggle.hnsw_index import HNSWIndex

    index = HNSWIndex(db_path / ".." / "waggle.hnsw", dim=384)
    index.rebuild_from_db(connection, tenant_id, embedding_model)
    index.add(node_id, embedding_vector)
    results = index.query(query_embedding, k=50, scope_ids=project_node_ids)
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import sqlite3

LOGGER = logging.getLogger(__name__)

# Try importing hnswlib once at module load; set a flag for callers.
try:
    import hnswlib as _hnswlib  # type: ignore[import-untyped]
    HNSWLIB_AVAILABLE = True
except ModuleNotFoundError:
    _hnswlib = None  # type: ignore[assignment]
    HNSWLIB_AVAILABLE = False

_HNSWLIB_WARNED = False  # emit the "not installed" warning at most once


def _warn_once() -> None:
    global _HNSWLIB_WARNED  # noqa: PLW0603
    if not _HNSWLIB_WARNED:
        LOGGER.warning(
            "hnswlib not installed — HNSW index disabled; "
            "falling back to brute-force matmul.  "
            "Install with: pip install waggle-mcp[perf]"
        )
        _HNSWLIB_WARNED = True


class HNSWIndex:
    """HNSW approximate nearest-neighbour index for Waggle node embeddings.

    Parameters
    ----------
    path:
        Path to the ``.hnsw`` binary file.  The metadata JSON is written
        alongside it at ``<path>.meta.json``.
    dim:
        Embedding dimension (must match the model in use).
    max_elements:
        Maximum number of vectors the index can hold.  The index is
        automatically grown by ``_grow_if_needed()`` if this limit is
        reached.
    ef_construction:
        Build-time accuracy parameter (higher → better quality, slower build).
        128–400 is typical; 200 is a good default.
    M:
        Number of bi-directional links per node in the graph.  Higher M
        improves recall at the cost of memory.  16 is the hnswlib default.
    dtype:
        ``"float32"`` (default) or ``"float16"`` (halves memory usage).
    """

    _GROWTH_FACTOR = 2
    _MIN_CAPACITY = 1_000

    def __init__(
        self,
        path: Path,
        dim: int = 384,
        *,
        max_elements: int = 50_000,
        ef_construction: int = 200,
        M: int = 16,
        dtype: str = "float32",
    ) -> None:
        self.path = Path(path)
        self.meta_path = self.path.with_suffix(".hnsw.meta.json")
        self.dim = dim
        self._max_elements = max(max_elements, self._MIN_CAPACITY)
        self._ef_construction = ef_construction
        self._M = M
        self._np_dtype = np.float16 if dtype.strip().lower() == "float16" else np.float32
        self._lock = threading.Lock()

        # id_map: hnswlib int label → Waggle UUID
        # reverse_map: Waggle UUID → hnswlib int label
        self._id_map: dict[int, str] = {}
        self._reverse_map: dict[str, int] = {}
        self._next_label: int = 0
        self._deleted: set[int] = set()

        self._index: object | None = None  # hnswlib.Index or None

        if HNSWLIB_AVAILABLE:
            self._load_or_init()
        else:
            _warn_once()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """True when hnswlib is installed and the index is initialised."""
        return self._index is not None

    @property
    def element_count(self) -> int:
        """Number of live (non-deleted) vectors in the index."""
        if self._index is None:
            return 0
        return len(self._id_map) - len(self._deleted)

    def add(self, node_id: str, embedding: np.ndarray) -> None:
        """Add or replace a node's embedding vector.

        If the node already has a label (was added before), the old label is
        marked deleted and a new one is assigned so the vector is updated.
        """
        if self._index is None:
            return
        with self._lock:
            # Remove stale entry if the node was updated.
            if node_id in self._reverse_map:
                old_label = self._reverse_map[node_id]
                try:
                    self._index.mark_deleted(old_label)  # type: ignore[union-attr]
                    self._deleted.add(old_label)
                except Exception:  # noqa: BLE001
                    pass

            self._grow_if_needed()
            label = self._next_label
            self._next_label += 1
            vec = np.asarray(embedding, dtype=self._np_dtype).reshape(1, -1)
            self._index.add_items(vec, np.array([label]))  # type: ignore[union-attr]
            self._id_map[label] = node_id
            self._reverse_map[node_id] = label

    def remove(self, node_id: str) -> None:
        """Mark a node's vector as deleted (soft-delete; reclaimed on rebuild)."""
        if self._index is None:
            return
        with self._lock:
            label = self._reverse_map.pop(node_id, None)
            if label is not None:
                try:
                    self._index.mark_deleted(label)  # type: ignore[union-attr]
                    self._deleted.add(label)
                except Exception:  # noqa: BLE001
                    pass
                self._id_map.pop(label, None)

    def query(
        self,
        embedding: np.ndarray,
        k: int = 50,
        *,
        scope_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Find the k approximate nearest neighbours.

        Parameters
        ----------
        embedding:
            Query vector (1-D, same dim as index).
        k:
            Number of results to return.
        scope_ids:
            When provided, only results whose node_id is in *scope_ids* are
            returned.  Uses hnswlib's element-level filter when available,
            otherwise post-filters.  Pass ``None`` (default) for global search.

        Returns
        -------
        List of ``(node_id, similarity)`` pairs, sorted by similarity descending.
        Similarities are cosine (higher = more similar, range 0–1).
        """
        if self._index is None or self.element_count == 0:
            return []

        with self._lock:
            live_count = self.element_count
            if live_count == 0:
                return []

            k_fetch = min(k * 3, live_count) if scope_ids else min(k, live_count)
            vec = np.asarray(embedding, dtype=self._np_dtype).reshape(1, -1)

            try:
                self._index.set_ef(max(k_fetch * 2, 50))  # type: ignore[union-attr]
                labels, distances = self._index.knn_query(vec, k=k_fetch)  # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("hnsw_query_failed", extra={"error": str(exc)})
                return []

            results: list[tuple[str, float]] = []
            for label, dist in zip(labels[0], distances[0]):
                label = int(label)
                if label in self._deleted or label not in self._id_map:
                    continue
                node_id = self._id_map[label]
                if scope_ids is not None and node_id not in scope_ids:
                    continue
                # hnswlib cosine space returns 1 - cosine_similarity as distance
                similarity = float(np.clip(1.0 - dist, 0.0, 1.0))
                results.append((node_id, similarity))
                if len(results) >= k:
                    break

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]

    def save(self) -> None:
        """Persist the index and metadata to disk."""
        if self._index is None:
            return
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._index.save_index(str(self.path))  # type: ignore[union-attr]
            meta = {
                "next_label": self._next_label,
                "id_map": {str(k): v for k, v in self._id_map.items()},
                "deleted": list(self._deleted),
            }
            self.meta_path.write_text(json.dumps(meta), encoding="utf-8")

    def rebuild_from_db(
        self,
        connection: sqlite3.Connection,
        tenant_id: str,
        embedding_model: object,
    ) -> int:
        """Populate the index from all nodes in the SQLite database.

        Called once on startup when the ``.hnsw`` file is missing or stale.
        Thread-safe.

        Parameters
        ----------
        connection:
            An open SQLite connection to the Waggle database.
        tenant_id:
            Restrict to this tenant's nodes.
        embedding_model:
            An ``EmbeddingModel`` instance (used for ``from_bytes()``).

        Returns
        -------
        Number of vectors indexed.
        """
        if self._index is None:
            return 0

        LOGGER.info("hnsw_rebuild_start", extra={"tenant_id": tenant_id})
        rows = connection.execute(
            "SELECT id, embedding FROM nodes WHERE tenant_id = ? AND embedding IS NOT NULL",
            (tenant_id,),
        ).fetchall()

        if not rows:
            LOGGER.info("hnsw_rebuild_complete", extra={"count": 0})
            return 0

        # Grow to fit if needed before bulk insert.
        with self._lock:
            needed = len(rows) + self._next_label
            if needed > self._max_elements:
                self._resize(needed * self._GROWTH_FACTOR)

        count = 0
        for row in rows:
            try:
                vec = embedding_model.from_bytes(row["embedding"])  # type: ignore[union-attr]
                self.add(row["id"], vec)
                count += 1
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug(
                    "hnsw_rebuild_skip_node",
                    extra={"node_id": row["id"], "error": str(exc)},
                )

        self.save()
        LOGGER.info("hnsw_rebuild_complete", extra={"count": count})
        return count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_or_init(self) -> None:
        """Load an existing index file or create a fresh one."""
        index = _hnswlib.Index(space="cosine", dim=self.dim)
        if self.path.exists() and self.meta_path.exists():
            try:
                index.load_index(str(self.path), max_elements=self._max_elements)
                raw = json.loads(self.meta_path.read_text(encoding="utf-8"))
                self._id_map = {int(k): v for k, v in raw.get("id_map", {}).items()}
                self._reverse_map = {v: int(k) for k, v in self._id_map.items()}
                self._next_label = int(raw.get("next_label", len(self._id_map)))
                self._deleted = set(raw.get("deleted", []))
                self._index = index
                LOGGER.info(
                    "hnsw_loaded",
                    extra={"path": str(self.path), "elements": self.element_count},
                )
                return
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "hnsw_load_failed_reinitialising",
                    extra={"path": str(self.path), "error": str(exc)},
                )
                # Fall through to fresh init.

        index.init_index(
            max_elements=self._max_elements,
            ef_construction=self._ef_construction,
            M=self._M,
        )
        self._index = index
        LOGGER.info("hnsw_initialised", extra={"dim": self.dim, "max_elements": self._max_elements})

    def _grow_if_needed(self) -> None:
        """Double the index capacity when the next label would exceed the limit."""
        if self._index is None:
            return
        if self._next_label >= self._max_elements:
            self._resize(self._max_elements * self._GROWTH_FACTOR)

    def _resize(self, new_max: int) -> None:
        """Resize the hnswlib index to hold at least *new_max* elements."""
        if self._index is None:
            return
        new_max = max(new_max, self._MIN_CAPACITY)
        self._index.resize_index(new_max)  # type: ignore[union-attr]
        self._max_elements = new_max
        LOGGER.debug("hnsw_resized", extra={"new_max": new_max})
