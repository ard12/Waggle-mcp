"""waggle/memory_lifecycle.py
==========================
Importance scoring and memory decay for Waggle.

Nodes accumulate importance from three sources:
  1. Base importance — determined by node type (decisions > facts > notes).
  2. Recency decay — importance decays exponentially since last access.
  3. Edge reinforcement — nodes with more typed edges decay more slowly.

The module is deliberately free of graph I/O; all inputs are plain Python
values so it can be unit-tested without a database.

Feature flag: WAGGLE_DECAY_ENABLED (bool, default false).
Half-life:    WAGGLE_DECAY_HALF_LIFE_DAYS (float, default 14.0).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from waggle.models import Node, NodeType

# ---------------------------------------------------------------------------
# Base importance by node type
# ---------------------------------------------------------------------------

# How "sticky" each node type is before any decay is applied.
# Decisions and preferences encode deliberate choices — they should persist
# longer.  Notes and questions are ephemeral and decay fastest.
_BASE_IMPORTANCE: dict[str, float] = {
    "decision":   1.00,
    "preference": 0.95,
    "concept":    0.90,
    "entity":     0.85,
    "fact":       0.75,
    "note":       0.60,
    "question":   0.55,
}

_DEFAULT_BASE: float = 0.70


def _base_importance(node_type: "NodeType | str") -> float:
    """Return the base importance weight for a given node type."""
    key = node_type.value if hasattr(node_type, "value") else str(node_type).lower()
    return _BASE_IMPORTANCE.get(key, _DEFAULT_BASE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_importance(
    node: "Node",
    *,
    now: datetime | None = None,
    half_life_days: float = 14.0,
    edge_count: int = 0,
) -> float:
    """Compute the current importance score for *node*.

    Parameters
    ----------
    node:
        The Node to score.  Must have ``node_type``, ``updated_at``,
        ``access_count``, and optionally ``last_accessed_at``.
    now:
        Reference time (UTC).  Defaults to ``datetime.now(timezone.utc)``.
    half_life_days:
        Decay half-life in days.  A node that has not been accessed in
        ``half_life_days`` days retains 50% of its base importance.
    edge_count:
        Number of typed edges connected to this node.  More edges slow decay
        (edge reinforcement).

    Returns
    -------
    float
        Importance in [0.0, 1.0].  A score of 1.0 means perfectly fresh and
        heavily reinforced.  A score approaching 0.0 means stale and isolated.
    """
    if half_life_days <= 0:
        half_life_days = 14.0

    now_dt = now or datetime.now(timezone.utc)

    # --- 1. Base importance by type ---
    base = _base_importance(node.node_type)

    # --- 2. Recency decay (exponential) ---
    # Use last_accessed_at when available; fall back to updated_at.
    last_touch: datetime = getattr(node, "last_accessed_at", None) or node.updated_at
    if last_touch.tzinfo is None:
        last_touch = last_touch.replace(tzinfo=timezone.utc)
    now_aware = now_dt.replace(tzinfo=timezone.utc) if now_dt.tzinfo is None else now_dt.astimezone(timezone.utc)
    age_days = max((now_aware - last_touch).total_seconds() / 86_400.0, 0.0)

    # Edge reinforcement: each edge adds a small fraction to the effective
    # half-life, capped at 4× the base half-life to avoid immortal nodes.
    effective_half_life = min(half_life_days * (1.0 + 0.1 * edge_count), half_life_days * 4.0)
    decay = math.pow(0.5, age_days / effective_half_life)

    # --- 3. Access frequency boost (log-dampened) ---
    access_boost = math.log1p(node.access_count) / math.log1p(50)  # saturates at ~50 accesses
    access_boost = min(access_boost, 0.25)  # cap contribution at 25%

    importance = base * decay + access_boost
    return min(max(importance, 0.0), 1.0)


def identify_prunable(
    nodes: "list[Node]",
    *,
    now: datetime | None = None,
    half_life_days: float = 14.0,
    importance_threshold: float = 0.05,
    min_age_days: float = 90.0,
    edge_counts: dict[str, int] | None = None,
) -> list[str]:
    """Return IDs of nodes eligible for pruning.

    A node is prunable when ALL of the following hold:
      - Its computed importance is below *importance_threshold*.
      - It was created at least *min_age_days* ago (never prune fresh nodes).
      - It has no active validity window (``valid_to`` is None or expired).

    Parameters
    ----------
    nodes:
        All candidate nodes (typically the full graph for a tenant).
    now:
        Reference time.  Defaults to ``datetime.now(timezone.utc)``.
    half_life_days:
        Decay half-life passed to :func:`compute_importance`.
    importance_threshold:
        Nodes scoring below this are considered prunable.
    min_age_days:
        Nodes younger than this are never returned, regardless of score.
    edge_counts:
        Mapping of ``node_id`` → number of connected edges.  When absent,
        edge reinforcement is skipped (conservative: prunes fewer nodes).

    Returns
    -------
    list[str]
        Node IDs (not Node objects) — the caller decides what to do with them.
    """
    now_dt = now or datetime.now(timezone.utc)
    now_aware = now_dt.replace(tzinfo=timezone.utc) if now_dt.tzinfo is None else now_dt.astimezone(timezone.utc)
    edge_counts = edge_counts or {}

    prunable: list[str] = []
    for node in nodes:
        # Age gate: never prune recently-created nodes
        created = node.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_days = (now_aware - created).total_seconds() / 86_400.0
        if age_days < min_age_days:
            continue

        # Skip nodes with an active (future) validity window — they are
        # intentionally scoped and should not be pruned automatically.
        if node.valid_to is not None:
            vt = node.valid_to.replace(tzinfo=timezone.utc) if node.valid_to.tzinfo is None else node.valid_to
            if vt > now_aware:
                continue  # still within its validity window

        score = compute_importance(
            node,
            now=now_dt,
            half_life_days=half_life_days,
            edge_count=edge_counts.get(node.id, 0),
        )
        if score < importance_threshold:
            prunable.append(node.id)

    return prunable
