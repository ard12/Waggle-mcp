"""Shared memory embedding cache for zero-copy cross-process retrieval.

Problem
-------
Every ``aggregate()`` and ``_query_graph_only()`` call fetches every node's
embedding BLOB from SQLite, then calls ``EmbeddingModel.from_bytes()`` to
deserialise it into a numpy array.  For 10 K nodes at 1536-dim float32 this
means transferring ~60 MB from SQLite *and* creating 10 K temporary numpy
arrays on every query — even when the embeddings haven't changed.

Solution
--------
``EmbeddingCache`` keeps a memory-mapped flat binary file (``waggle.emb``)
alongside ``waggle.db``.  The file holds all embeddings as a contiguous
``(capacity, dim)`` float32 matrix.  Reads are zero-copy views into the
mmap'd region; the OS page cache automatically deduplicates the physical
memory across every process that maps the same file.

``SharedMemoryBridge`` goes one step further: it copies the matrix into a
named ``multiprocessing.shared_memory`` segment.  Any process that knows the
name can attach and get a direct numpy view — no file I/O, no deserialization.

File layout (``waggle.emb``)
-----------------------------
    Offset 0: Header (64 bytes, big-endian)
        magic    [8s]  b"WAGLEMB1"
        version  [H]   format version (currently 1)
        dtype    [H]   0 = float32 | 1 = float16
        dim      [I]   embedding dimension
        capacity [Q]   maximum rows allocated in the matrix region
        n_nodes  [Q]   rows currently written
        reserved [32s] padding to 64 bytes

    Offset 64: Embedding matrix
        capacity × dim × itemsize bytes
        row i = embedding for the node whose ID is recorded in the index file

Index file (``waggle.emb.idx``)
--------------------------------
JSON  {
    "version": 1,
    "next_row": <int>,          # next free row to write
    "node_row": {"<uuid>": row_index, ...},
    "deleted_rows": [row_idx, ...]
}

Usage
-----
    cache = EmbeddingCache(db_path.parent / "waggle.emb", dim=384)
    loaded = cache.load()

    if not loaded:
        with connect(db_path) as conn:
            cache.rebuild_from_db(conn, tenant_id, embedding_model)

    # Read path (zero-copy):
    vec = cache.get("some-node-uuid")       # np.ndarray view | None

    # Write path (after add_node):
    cache.put("some-node-uuid", embedding_vector)
    cache.flush_index()                     # persist the JSON index

    # Integration in aggregate():
    emb_matrix, ids = cache.load_matrix(candidate_ids)   # batch fetch
"""

from __future__ import annotations

import json
import logging
import mmap
import os
import struct
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import sqlite3

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Binary header constants
# ---------------------------------------------------------------------------
_MAGIC = b"WAGLEMB1"
_FORMAT_VERSION = 1
_HEADER_STRUCT = struct.Struct(">8sHHIQQ32s")  # 8+2+2+4+8+8+32 = 64 bytes
_HEADER_SIZE = _HEADER_STRUCT.size              # exactly 64

assert _HEADER_SIZE == 64, f"Expected 64-byte header, got {_HEADER_SIZE}"

_DTYPE_CODE = {np.dtype("float32"): 0, np.dtype("float16"): 1}
_CODE_DTYPE = {0: np.dtype("float32"), 1: np.dtype("float16")}

_IDX_VERSION = 1
_GROWTH_FACTOR = 2
_MIN_CAPACITY = 512

# ---------------------------------------------------------------------------
# SharedMemoryBridge (optional cross-process zero-copy)
# ---------------------------------------------------------------------------

try:
    from multiprocessing.shared_memory import SharedMemory as _SharedMemory
    _SHM_AVAILABLE = True
except ImportError:  # Python < 3.8 (shouldn't happen on 3.14)
    _SharedMemory = None  # type: ignore[assignment,misc]
    _SHM_AVAILABLE = False


class SharedMemoryBridge:
    """Copy an embedding matrix into a named OS shared-memory segment.

    All processes that know the *name* can attach and get a numpy array view
    into the same physical pages — truly zero-copy, zero file I/O after the
    first ``create()`` call.

    Notes
    -----
    * The creator process must keep the bridge alive (i.e., do not call
      ``close()`` while other processes may still be reading).
    * The named segment persists until ``unlink()`` is called, even after
      the creator exits.
    * Windows: ``SharedMemory`` uses named file-mapping objects, which are
      automatically cleaned up when all handles are closed.  Explicit
      ``unlink()`` is a no-op on Windows but harmless.
    """

    def __init__(self, name: str, shape: tuple[int, int], dtype: np.dtype) -> None:
        self.name = name
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self._shm: object | None = None
        self._array: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        return _SHM_AVAILABLE

    @property
    def array(self) -> np.ndarray | None:
        return self._array

    def create(self, matrix: np.ndarray) -> bool:
        """Allocate a new shared-memory segment and copy *matrix* into it.

        Returns True on success, False if ``shared_memory`` is not available.
        """
        if not _SHM_AVAILABLE:
            LOGGER.debug("shared_memory_not_available_skipping_bridge")
            return False
        nbytes = int(np.prod(matrix.shape)) * matrix.dtype.itemsize
        try:
            self._shm = _SharedMemory(name=self.name, create=True, size=nbytes)
            arr = np.ndarray(matrix.shape, dtype=matrix.dtype, buffer=self._shm.buf)  # type: ignore[union-attr]
            arr[:] = matrix
            self._array = arr
            LOGGER.info("shm_bridge_created", extra={"name": self.name, "bytes": nbytes})
            return True
        except FileExistsError:
            # Segment already exists from a previous run — attach instead.
            return self.attach()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("shm_bridge_create_failed", extra={"error": str(exc)})
            return False

    def attach(self) -> bool:
        """Attach to an existing shared-memory segment created by another process."""
        if not _SHM_AVAILABLE:
            return False
        try:
            self._shm = _SharedMemory(name=self.name, create=False)
            self._array = np.ndarray(
                self.shape, dtype=self.dtype, buffer=self._shm.buf  # type: ignore[union-attr]
            )
            LOGGER.info("shm_bridge_attached", extra={"name": self.name})
            return True
        except FileNotFoundError:
            LOGGER.debug("shm_bridge_not_found", extra={"name": self.name})
            return False
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("shm_bridge_attach_failed", extra={"error": str(exc)})
            return False

    def close(self) -> None:
        """Release the handle to the shared-memory segment (does not free it)."""
        if self._shm is not None:
            try:
                self._shm.close()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass
            self._shm = None
            self._array = None

    def unlink(self) -> None:
        """Destroy the shared-memory segment (call only from the creator)."""
        if not _SHM_AVAILABLE:
            return
        try:
            shm = _SharedMemory(name=self.name, create=False)
            shm.close()
            shm.unlink()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# EmbeddingCache
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """mmap-backed flat binary embedding cache.

    Parameters
    ----------
    path:
        Path to the ``waggle.emb`` file.  The index is stored alongside
        at ``<path>.idx``.
    dim:
        Embedding dimension.  Must match the model in use.
    dtype:
        ``"float32"`` (default) or ``"float16"`` (halves file size).
    initial_capacity:
        Number of embedding slots to pre-allocate in the binary file.
        The cache grows automatically when full.
    """

    def __init__(
        self,
        path: Path,
        dim: int = 384,
        *,
        dtype: str = "float32",
        initial_capacity: int = _MIN_CAPACITY,
    ) -> None:
        self.path = Path(path)
        self.idx_path = Path(str(path) + ".idx")
        self.dim = dim
        self._np_dtype: np.dtype = np.dtype("float16") if dtype.strip().lower() == "float16" else np.dtype("float32")
        self._dtype_code = _DTYPE_CODE.get(self._np_dtype, 0)
        self._initial_capacity = max(initial_capacity, _MIN_CAPACITY)
        self._lock = threading.RLock()

        # Mmap state
        self._mm: mmap.mmap | None = None
        self._file: object | None = None        # file object kept open for mmap
        self._capacity: int = 0
        self._n_nodes: int = 0

        # Index state
        self._node_row: dict[str, int] = {}     # node_id → row index
        self._deleted_rows: set[int] = set()
        self._next_row: int = 0

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._mm is not None

    @property
    def cached_count(self) -> int:
        return len(self._node_row) - len(self._deleted_rows)

    def get(self, node_id: str) -> np.ndarray | None:
        """Return a zero-copy numpy view of the node's embedding.

        Returns None if the node is not in the cache.  The returned array
        shares memory with the mmap region — do not modify it in place.
        """
        if self._mm is None:
            return None
        row = self._node_row.get(node_id)
        if row is None or row in self._deleted_rows:
            return None
        with self._lock:
            offset = _HEADER_SIZE + row * self.dim * self._np_dtype.itemsize
            try:
                return np.frombuffer(
                    self._mm, dtype=self._np_dtype, count=self.dim, offset=offset
                )
            except (ValueError, OSError):
                return None

    def load_matrix(
        self, node_ids: list[str]
    ) -> tuple[np.ndarray | None, list[str]]:
        """Return a stacked matrix of embeddings for *node_ids* found in cache.

        Returns
        -------
        matrix:
            ``(M, dim)`` float32 array for the M cache-hit IDs, or None.
        hit_ids:
            The subset of *node_ids* that were found in the cache,
            in the same order as matrix rows.
        """
        if self._mm is None or not self._node_row:
            return None, []
        vecs: list[np.ndarray] = []
        hit_ids: list[str] = []
        for nid in node_ids:
            vec = self.get(nid)
            if vec is not None:
                vecs.append(vec.astype(np.float32, copy=False))
                hit_ids.append(nid)
        if not vecs:
            return None, []
        return np.stack(vecs), hit_ids

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def put(self, node_id: str, embedding: np.ndarray) -> None:
        """Write or update a node's embedding in the cache.

        Thread-safe.  If the node already exists, the old slot is marked
        deleted and a new row is appended (compacted on next rebuild).
        """
        if self._mm is None:
            return
        with self._lock:
            # Soft-delete old slot if node is being re-inserted.
            old_row = self._node_row.get(node_id)
            if old_row is not None:
                self._deleted_rows.add(old_row)

            if self._next_row >= self._capacity:
                self._grow(self._capacity * _GROWTH_FACTOR)

            row = self._next_row
            self._next_row += 1
            self._n_nodes += 1

            vec = np.asarray(embedding, dtype=self._np_dtype)
            offset = _HEADER_SIZE + row * self.dim * self._np_dtype.itemsize
            self._mm.seek(offset)
            self._mm.write(vec.tobytes())

            self._node_row[node_id] = row
            self._update_header()

    def remove(self, node_id: str) -> None:
        """Soft-delete a node from the cache (slot reclaimed on next rebuild)."""
        with self._lock:
            row = self._node_row.pop(node_id, None)
            if row is not None:
                self._deleted_rows.add(row)

    def flush_index(self) -> None:
        """Persist the JSON index to disk.  Call after a batch of puts/removes."""
        with self._lock:
            idx = {
                "version": _IDX_VERSION,
                "next_row": self._next_row,
                "node_row": self._node_row,
                "deleted_rows": list(self._deleted_rows),
            }
        self.idx_path.write_text(json.dumps(idx), encoding="utf-8")

    # ------------------------------------------------------------------
    # Open / close / load / rebuild
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """Open or create the binary file and memory-map it.

        Returns True if the cache is usable, False on error.
        """
        try:
            if self.path.exists():
                return self._open_existing()
            else:
                return self._create_new(self._initial_capacity)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("emb_cache_open_failed", extra={"error": str(exc)})
            return False

    def close(self) -> None:
        """Flush and close the mmap file."""
        with self._lock:
            if self._mm is not None:
                try:
                    self._mm.flush()
                    self._mm.close()
                except Exception:  # noqa: BLE001
                    pass
                self._mm = None
            if self._file is not None:
                try:
                    self._file.close()  # type: ignore[union-attr]
                except Exception:  # noqa: BLE001
                    pass
                self._file = None

    def load(self) -> bool:
        """Open the mmap file and load the JSON index.

        Returns True if both exist and are readable.
        """
        if not self.path.exists() or not self.idx_path.exists():
            return False
        ok = self.open()
        if not ok:
            return False
        try:
            raw = json.loads(self.idx_path.read_text(encoding="utf-8"))
            if raw.get("version") != _IDX_VERSION:
                LOGGER.warning("emb_cache_index_version_mismatch")
                return False
            with self._lock:
                self._node_row = {k: int(v) for k, v in raw.get("node_row", {}).items()}
                self._deleted_rows = set(int(r) for r in raw.get("deleted_rows", []))
                self._next_row = int(raw.get("next_row", len(self._node_row)))
            LOGGER.info(
                "emb_cache_loaded",
                extra={"nodes": self.cached_count, "path": str(self.path)},
            )
            return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("emb_cache_index_load_failed", extra={"error": str(exc)})
            return False

    def rebuild_from_db(
        self,
        connection: sqlite3.Connection,
        tenant_id: str,
        embedding_model: object,
    ) -> int:
        """Populate the cache from all nodes in the SQLite database.

        Parameters
        ----------
        connection:
            Open ``sqlite3.Connection`` to the Waggle database.
        tenant_id:
            Restrict to this tenant's nodes.
        embedding_model:
            ``EmbeddingModel`` instance used for ``from_bytes()``.

        Returns
        -------
        Number of embeddings written to cache.
        """
        LOGGER.info("emb_cache_rebuild_start", extra={"tenant_id": tenant_id})
        rows = connection.execute(
            "SELECT id, embedding FROM nodes WHERE tenant_id = ? AND embedding IS NOT NULL",
            (tenant_id,),
        ).fetchall()

        if not rows:
            LOGGER.info("emb_cache_rebuild_complete", extra={"count": 0})
            return 0

        # Ensure capacity before bulk write.
        if not self.is_open:
            self._create_new(max(len(rows) * _GROWTH_FACTOR, _MIN_CAPACITY))

        count = 0
        for row in rows:
            try:
                vec = embedding_model.from_bytes(row["embedding"])  # type: ignore[union-attr]
                self.put(row["id"], vec)
                count += 1
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("emb_cache_skip_node", extra={"id": row["id"], "error": str(exc)})

        self.flush_index()
        LOGGER.info("emb_cache_rebuild_complete", extra={"count": count})
        return count

    def make_shared_bridge(self, name: str) -> SharedMemoryBridge:
        """Return a SharedMemoryBridge loaded with the current embedding matrix.

        The bridge copies all cached embeddings into a named OS shared-memory
        segment so other processes can attach by name without file I/O.

        Parameters
        ----------
        name:
            Name of the shared-memory segment (e.g. ``"waggle-emb-default"``).
        """
        bridge = SharedMemoryBridge(
            name=name,
            shape=(self._next_row, self.dim),
            dtype=self._np_dtype,
        )
        if not bridge.is_available:
            return bridge
        # Build a dense matrix from all live slots.
        with self._lock:
            if self._mm is None or self._next_row == 0:
                return bridge
            total_bytes = self._next_row * self.dim * self._np_dtype.itemsize
            raw = self._mm[_HEADER_SIZE: _HEADER_SIZE + total_bytes]
            matrix = np.frombuffer(raw, dtype=self._np_dtype).reshape(self._next_row, self.dim).copy()
        bridge.create(matrix)
        return bridge

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_new(self, capacity: int) -> bool:
        """Create a fresh binary file and mmap it."""
        capacity = max(capacity, _MIN_CAPACITY)
        itemsize = self._np_dtype.itemsize
        file_size = _HEADER_SIZE + capacity * self.dim * itemsize
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Pre-allocate file by writing zeros so mmap has a valid backing.
            with open(self.path, "wb") as f:
                f.write(b"\x00" * file_size)
            f = open(self.path, "r+b")  # reopen for r+b mmap
            self._file = f
            self._mm = mmap.mmap(f.fileno(), file_size)
            self._capacity = capacity
            self._n_nodes = 0
            self._next_row = 0
            self._write_header()
            LOGGER.info("emb_cache_created", extra={"capacity": capacity, "dim": self.dim})
            return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("emb_cache_create_failed", extra={"error": str(exc)})
            return False

    def _open_existing(self) -> bool:
        """Open and validate an existing binary file, then mmap it."""
        try:
            f = open(self.path, "r+b")  # noqa: WPS515
            raw_header = f.read(_HEADER_SIZE)
            magic, version, dtype_code, dim, capacity, n_nodes, _ = _HEADER_STRUCT.unpack(raw_header)
            if magic != _MAGIC:
                LOGGER.warning("emb_cache_bad_magic", extra={"path": str(self.path)})
                f.close()
                return False
            if version != _FORMAT_VERSION:
                LOGGER.warning("emb_cache_version_mismatch", extra={"version": version})
                f.close()
                return False
            self._file = f
            file_size = _HEADER_SIZE + int(capacity) * int(dim) * _CODE_DTYPE.get(dtype_code, np.float32).itemsize
            f.seek(0)
            self._mm = mmap.mmap(f.fileno(), file_size)
            self._capacity = int(capacity)
            self._n_nodes = int(n_nodes)
            self.dim = int(dim)
            self._np_dtype = _CODE_DTYPE.get(int(dtype_code), np.dtype("float32"))
            LOGGER.info("emb_cache_opened", extra={"capacity": capacity, "n_nodes": n_nodes})
            return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("emb_cache_open_existing_failed", extra={"error": str(exc)})
            return False

    def _write_header(self) -> None:
        """Write the 64-byte header to the start of the mmap."""
        header = _HEADER_STRUCT.pack(
            _MAGIC,
            _FORMAT_VERSION,
            self._dtype_code,
            self.dim,
            self._capacity,
            self._n_nodes,
            b"\x00" * 32,
        )
        self._mm.seek(0)  # type: ignore[union-attr]
        self._mm.write(header)  # type: ignore[union-attr]

    def _update_header(self) -> None:
        """Update only the mutable fields (n_nodes) in the header."""
        # Patch n_nodes at offset 20 (magic8 + version2 + dtype2 + dim4 = 16, + capacity8 = 24 → n_nodes at 24)
        self._mm.seek(24)  # type: ignore[union-attr]
        self._mm.write(struct.pack(">Q", self._n_nodes))  # type: ignore[union-attr]

    def _grow(self, new_capacity: int) -> None:
        """Extend the binary file and remap."""
        new_capacity = max(new_capacity, self._capacity + _MIN_CAPACITY)
        itemsize = self._np_dtype.itemsize
        new_size = _HEADER_SIZE + new_capacity * self.dim * itemsize

        LOGGER.debug("emb_cache_growing", extra={"old": self._capacity, "new": new_capacity})

        # Close the current mmap before resizing the file.
        if self._mm is not None:
            self._mm.flush()
            self._mm.close()
            self._mm = None

        # Extend the file with zeros up to the new size.
        self._file.seek(0, 2)  # type: ignore[union-attr]  # seek to EOF
        current_size = self._file.tell()  # type: ignore[union-attr]
        extra = new_size - current_size
        if extra > 0:
            self._file.write(b"\x00" * extra)  # type: ignore[union-attr]
            self._file.flush()  # type: ignore[union-attr]

        # Remap with the new file size.
        self._mm = mmap.mmap(self._file.fileno(), new_size)  # type: ignore[union-attr]
        self._capacity = new_capacity
        self._write_header()
