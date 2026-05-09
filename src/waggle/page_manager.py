"""Spatial graph paging — MemPalace-inspired locality clustering.

Maps Waggle's flat node graph onto a two-level spatial hierarchy:
  Palace  → Tenant   (top-level isolation, unchanged)
  Wing    → Project  (per-project partition — first-class scope)
  Room    → Page     (K-means cluster within a project)
  Drawer  → Node     (individual memory node)

This mirrors MemPalace's wing/room/drawer model but implemented over
Waggle's existing (tenant, project, session) scoping rather than requiring
a separate database schema.

Key capabilities
----------------
1. **Page-centroid scoring** (primary integration point)
   Before scoring N individual nodes, score P page centroids (P << N).
   This gives a fast "neighborhood relevance" signal that boosts all nodes
   in pages whose centroids are close to the query — equivalent to
   MemPalace's "route to the right wing/room first" step.

2. **Co-page prefetch hints**
   After HNSW (or brute-force) returns top-K node IDs, expand the
   candidate set by adding their co-paged neighbours.  Nodes that share a
   page are likely to be semantically related and co-returned in queries.

3. **Incremental assignment**
   New nodes are assigned to the nearest existing centroid (no full
   rebuild required for each write).  A rebuild is triggered automatically
   when the orphan count exceeds a configurable threshold.

Usage
-----
    from waggle.page_manager import GraphPageManager

    pm = GraphPageManager(db_path.parent / "waggle.pages.json", page_size=128)
    built = pm.load()               # try to load from disk
    if not built:
        pm.build_pages(embeddings, projects)   # embeddings: {node_id: vec}
        pm.save()                              # projects:   {node_id: project}

    # Query path (called once per aggregate/query_graph call):
    page_scores = pm.score_pages(query_embedding)   # {page_id: float}

    # Per-node boost (called inside the scoring loop):
    boost = pm.get_page_boost(node_id, page_scores)   # float in [0, PAGE_BOOST_CAP]

    # Write path (called in add_node after SQLite insert):
    pm.assign_new_node(node_id, embedding, project)
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Iterable

import numpy as np

LOGGER = logging.getLogger(__name__)

# Maximum additive boost applied to a node's cosine score when its page
# centroid is the closest to the query.  0.15 keeps it below the minimum
# meaningful cosine delta while still influencing tie-breaking.
PAGE_BOOST_CAP: float = 0.15

# Minimum nodes needed before K-means clustering is worthwhile.
_MIN_NODES_FOR_CLUSTERING = 10

# Persistence schema version — bump if the JSON format changes.
_FORMAT_VERSION = 1


class GraphPageManager:
    """Two-level spatial clustering: project-partition → K-means rooms.

    Parameters
    ----------
    path:
        Path to the ``.pages.json`` persistence file.
    page_size:
        Target number of nodes per page (cluster).  Actual sizes may vary
        (±50%) depending on cluster geometry.
    rebuild_threshold:
        Trigger a rebuild when the number of unassigned (orphan) nodes
        exceeds this value.  Set to 0 to disable automatic rebuilds.
    """

    def __init__(
        self,
        path: Path,
        *,
        page_size: int = 128,
        rebuild_threshold: int = 500,
    ) -> None:
        self.path = Path(path)
        self.page_size = max(page_size, 8)
        self.rebuild_threshold = rebuild_threshold
        self._lock = threading.Lock()

        # Core state — populated by build_pages() or load()
        self._pages: dict[int, list[str]] = {}          # page_id → [node_ids]
        self._node_page: dict[str, int] = {}            # node_id → page_id
        self._centroids: np.ndarray | None = None        # (P, D) centroid matrix
        self._centroid_page_ids: list[int] = []         # centroid row → page_id
        self._node_count_at_build: int = 0
        self._orphan_count: int = 0

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    @property
    def is_built(self) -> bool:
        """True once pages have been built or loaded from disk."""
        return self._centroids is not None and len(self._node_page) > 0

    @property
    def page_count(self) -> int:
        return len(self._pages)

    @property
    def node_count(self) -> int:
        return len(self._node_page)

    def score_pages(self, query_embedding: np.ndarray) -> dict[int, float]:
        """Score all page centroids against the query embedding.

        Returns a dict mapping page_id → similarity (0–1, normalised
        cosine).  This is the cheap "wing routing" step: P centroid
        dot-products instead of N per-node cosine calls.

        When pages are not built, returns an empty dict (no boost applied).
        """
        if self._centroids is None or self._centroids.shape[0] == 0:
            return {}
        q = np.asarray(query_embedding, dtype=np.float32)
        # centroids are L2-normalised at build time
        sims: np.ndarray = np.clip(self._centroids @ q, 0.0, 1.0)
        return {
            self._centroid_page_ids[i]: float(sims[i])
            for i in range(len(self._centroid_page_ids))
        }

    def get_page_boost(
        self,
        node_id: str,
        page_scores: dict[int, float],
        *,
        boost_scale: float = PAGE_BOOST_CAP,
    ) -> float:
        """Return additive score boost for *node_id* given pre-computed page scores.

        The boost is proportional to the page's centroid similarity,
        capped at *boost_scale* (default 0.15).  Nodes in irrelevant pages
        receive 0.0 boost.
        """
        if not page_scores:
            return 0.0
        page_id = self._node_page.get(node_id)
        if page_id is None:
            return 0.0
        raw = page_scores.get(page_id, 0.0)
        return float(np.clip(raw * boost_scale, 0.0, boost_scale))

    def get_co_paged_nodes(self, node_ids: Iterable[str]) -> set[str]:
        """Return the union of pages containing *node_ids*, minus the input set.

        Use this to expand a candidate set with semantically co-located nodes.
        For a set of HNSW hits, this adds their page-neighbours — the nodes
        most likely to be relevant but not ranked in the top-K by the ANN query.
        """
        expanded: set[str] = set()
        seen_pages: set[int] = set()
        for nid in node_ids:
            pid = self._node_page.get(nid)
            if pid is not None and pid not in seen_pages:
                seen_pages.add(pid)
                expanded.update(self._pages.get(pid, []))
        # Subtract the original input to return only the newly added nodes.
        return expanded - set(node_ids)

    def rebuild_needed(self, current_node_count: int) -> bool:
        """Return True if the page structure is stale and should be rebuilt."""
        if not self.is_built:
            return True
        orphan_ratio_exceeded = (
            self.rebuild_threshold > 0
            and self._orphan_count >= self.rebuild_threshold
        )
        growth_ratio_exceeded = current_node_count > self._node_count_at_build * 1.5
        return orphan_ratio_exceeded or growth_ratio_exceeded

    # ------------------------------------------------------------------
    # Build & write API
    # ------------------------------------------------------------------

    def build_pages(
        self,
        embeddings: dict[str, np.ndarray],
        projects: dict[str, str],
    ) -> None:
        """Cluster nodes into pages.

        First partitions by project (MemPalace wing), then applies K-means
        within each project to form rooms (pages).  Both partitioning steps
        run entirely in numpy — no scipy/sklearn dependency.

        Parameters
        ----------
        embeddings:
            Mapping node_id → L2-normalised float32 embedding vector.
        projects:
            Mapping node_id → project string.
        """
        if not embeddings:
            LOGGER.info("page_manager_build_skip_empty")
            return

        LOGGER.info("page_manager_build_start", extra={"n_nodes": len(embeddings)})

        with self._lock:
            new_pages: dict[int, list[str]] = {}
            new_node_page: dict[str, int] = {}
            new_centroids_list: list[np.ndarray] = []
            new_centroid_page_ids: list[int] = []
            page_counter = 0

            # Group node IDs by project (Wing partitioning)
            by_project: dict[str, list[str]] = {}
            for node_id in embeddings:
                proj = projects.get(node_id, "")
                by_project.setdefault(proj, []).append(node_id)

            for proj, node_ids in sorted(by_project.items()):
                n = len(node_ids)
                if n < _MIN_NODES_FOR_CLUSTERING:
                    # Tiny project → single page; no clustering needed
                    new_pages[page_counter] = list(node_ids)
                    for nid in node_ids:
                        new_node_page[nid] = page_counter
                    centroid = _safe_mean([embeddings[nid] for nid in node_ids])
                    new_centroids_list.append(centroid)
                    new_centroid_page_ids.append(page_counter)
                    page_counter += 1
                else:
                    n_clusters = max(1, round(n / self.page_size))
                    matrix = np.stack([embeddings[nid] for nid in node_ids])
                    labels, centroids = _kmeans_lloyd(matrix, n_clusters)
                    for k in range(n_clusters):
                        cluster_nodes = [node_ids[i] for i, lbl in enumerate(labels) if lbl == k]
                        if not cluster_nodes:
                            continue
                        new_pages[page_counter] = cluster_nodes
                        for nid in cluster_nodes:
                            new_node_page[nid] = page_counter
                        new_centroids_list.append(centroids[k])
                        new_centroid_page_ids.append(page_counter)
                        page_counter += 1

            self._pages = new_pages
            self._node_page = new_node_page
            self._centroids = (
                np.stack(new_centroids_list).astype(np.float32)
                if new_centroids_list
                else None
            )
            self._centroid_page_ids = new_centroid_page_ids
            self._node_count_at_build = len(embeddings)
            self._orphan_count = 0

        LOGGER.info(
            "page_manager_build_complete",
            extra={"pages": page_counter, "nodes": len(embeddings)},
        )

    def assign_new_node(
        self,
        node_id: str,
        embedding: np.ndarray,
        project: str = "",
    ) -> int | None:
        """Assign a newly-written node to the nearest existing page centroid.

        This is an incremental update — no full rebuild required.
        Returns the assigned page_id, or None if no pages exist yet.
        """
        if self._centroids is None or self._centroids.shape[0] == 0:
            self._orphan_count += 1
            return None

        with self._lock:
            q = np.asarray(embedding, dtype=np.float32)
            sims = self._centroids @ q
            best_idx = int(np.argmax(sims))
            page_id = self._centroid_page_ids[best_idx]

            # Remove old assignment if the node is being re-inserted.
            old_page = self._node_page.get(node_id)
            if old_page is not None and old_page in self._pages:
                try:
                    self._pages[old_page].remove(node_id)
                except ValueError:
                    pass

            self._pages.setdefault(page_id, []).append(node_id)
            self._node_page[node_id] = page_id
            return page_id

    def remove_node(self, node_id: str) -> None:
        """Remove a node from its page (called when a node is deleted)."""
        with self._lock:
            page_id = self._node_page.pop(node_id, None)
            if page_id is not None and page_id in self._pages:
                try:
                    self._pages[page_id].remove(node_id)
                except ValueError:
                    pass

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist page assignments and centroids to disk."""
        if not self.is_built:
            return
        with self._lock:
            data: dict = {
                "version": _FORMAT_VERSION,
                "page_size": self.page_size,
                "node_count_at_build": self._node_count_at_build,
                "pages": {str(k): v for k, v in self._pages.items()},
                "node_page": {k: v for k, v in self._node_page.items()},
                "centroid_page_ids": self._centroid_page_ids,
                "centroids": (
                    self._centroids.tolist()
                    if self._centroids is not None
                    else None
                ),
            }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data), encoding="utf-8")
        LOGGER.info("page_manager_saved", extra={"path": str(self.path)})

    def load(self) -> bool:
        """Load page structure from disk.

        Returns True if loaded successfully, False otherwise.
        """
        if not self.path.exists():
            return False
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if raw.get("version") != _FORMAT_VERSION:
                LOGGER.warning("page_manager_version_mismatch", extra={"path": str(self.path)})
                return False
            with self._lock:
                self._pages = {int(k): v for k, v in raw["pages"].items()}
                self._node_page = raw["node_page"]
                self._centroid_page_ids = raw["centroid_page_ids"]
                raw_centroids = raw.get("centroids")
                self._centroids = (
                    np.array(raw_centroids, dtype=np.float32)
                    if raw_centroids
                    else None
                )
                self._node_count_at_build = raw.get("node_count_at_build", len(self._node_page))
                self._orphan_count = 0
            LOGGER.info(
                "page_manager_loaded",
                extra={"pages": len(self._pages), "nodes": len(self._node_page)},
            )
            return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("page_manager_load_failed", extra={"error": str(exc)})
            return False

    def rebuild_from_db(
        self,
        connection: object,
        tenant_id: str,
        embedding_model: object,
    ) -> int:
        """Build pages from all nodes currently in the SQLite database.

        Parameters
        ----------
        connection:
            An open ``sqlite3.Connection`` to the Waggle database.
        tenant_id:
            Scope the rebuild to this tenant's nodes.
        embedding_model:
            An ``EmbeddingModel`` instance used for ``from_bytes()``.

        Returns
        -------
        Number of nodes clustered into pages.
        """
        LOGGER.info("page_manager_rebuild_start", extra={"tenant_id": tenant_id})
        rows = connection.execute(  # type: ignore[union-attr]
            """
            SELECT id, project, embedding
            FROM nodes
            WHERE tenant_id = ? AND embedding IS NOT NULL
            """,
            (tenant_id,),
        ).fetchall()

        if not rows:
            LOGGER.info("page_manager_rebuild_complete", extra={"count": 0})
            return 0

        embeddings: dict[str, np.ndarray] = {}
        projects: dict[str, str] = {}
        for row in rows:
            try:
                vec = embedding_model.from_bytes(row["embedding"])  # type: ignore[union-attr]
                embeddings[row["id"]] = vec
                projects[row["id"]] = row["project"] or ""
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("page_manager_skip_node", extra={"id": row["id"], "error": str(exc)})

        self.build_pages(embeddings, projects)
        self.save()
        LOGGER.info("page_manager_rebuild_complete", extra={"count": len(embeddings)})
        return len(embeddings)


# ---------------------------------------------------------------------------
# Internal: pure-numpy K-means (Lloyd's algorithm + K-means++ init)
# ---------------------------------------------------------------------------

def _safe_mean(vecs: list[np.ndarray]) -> np.ndarray:
    """L2-normalised mean of a list of vectors."""
    m = np.mean(np.stack(vecs).astype(np.float32), axis=0)
    norm = float(np.linalg.norm(m))
    return m / norm if norm > 0.0 else m


def _kmeans_lloyd(
    matrix: np.ndarray,
    n_clusters: int,
    *,
    max_iter: int = 30,
    tol: float = 1e-4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's K-means with K-means++ initialisation.

    Parameters
    ----------
    matrix:
        (N, D) float32 matrix of L2-normalised embeddings.
    n_clusters:
        Number of clusters to form.
    max_iter:
        Maximum iterations for the Lloyd update loop.
    tol:
        Centroid movement tolerance for early convergence.
    seed:
        RNG seed for reproducible initialisation.

    Returns
    -------
    labels:    (N,) int32 cluster assignment for each row.
    centroids: (K, D) float32 L2-normalised centroid matrix.
    """
    n, d = matrix.shape
    n_clusters = min(n_clusters, n)

    if n_clusters == 1:
        return np.zeros(n, dtype=np.int32), _safe_mean(list(matrix)).reshape(1, -1)

    rng = np.random.default_rng(seed)

    # --- K-means++ initialisation ---
    # Choose first center uniformly at random, then each subsequent center
    # with probability proportional to squared distance to the nearest
    # already-chosen center.
    center_indices: list[int] = [int(rng.integers(n))]
    for _ in range(n_clusters - 1):
        chosen = np.stack([matrix[ci] for ci in center_indices])
        # For normalised vectors, cosine similarity = dot product.
        # Distance² = 2(1 - cos) for unit vectors.
        sims = matrix @ chosen.T        # (N, K_chosen)
        nearest_sim = sims.max(axis=1)  # (N,) — best similarity to any center
        dist_sq = np.clip(2.0 * (1.0 - nearest_sim), 0.0, None)
        total = float(dist_sq.sum())
        probs = dist_sq / total if total > 0 else np.ones(n) / n
        center_indices.append(int(rng.choice(n, p=probs)))

    centroids = matrix[center_indices].copy().astype(np.float32)

    # --- Lloyd's update loop ---
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        # Assignment step: each point → nearest centroid (max cosine similarity)
        sims = matrix @ centroids.T     # (N, K)
        new_labels = np.argmax(sims, axis=1).astype(np.int32)

        # Check convergence (no label changes)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Update step: recompute centroids as L2-normalised cluster means
        new_centroids = np.zeros_like(centroids)
        for k in range(n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                mean = matrix[mask].mean(axis=0)
                norm = float(np.linalg.norm(mean))
                new_centroids[k] = mean / norm if norm > 0.0 else mean
            else:
                new_centroids[k] = centroids[k]  # keep if cluster emptied

        # Convergence check on centroid movement
        if np.allclose(centroids, new_centroids, atol=tol):
            break
        centroids = new_centroids

    return labels, centroids
