"""Tests for MemoryGraph.graph_health() and _compute_health_score()."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pytest

from waggle.graph import MemoryGraph
from waggle.models import NodeType, RelationType


# ---------------------------------------------------------------------------
# Helpers — FakeEmbeddingModel (same pattern as test_edges.py)
# ---------------------------------------------------------------------------

class FakeEmbeddingModel:
    model_name = "fake-model"
    model_id = "fake-model:deterministic-v1"

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(8, dtype=np.float32)
        for token in text.lower().split():
            index = sum(ord(c) for c in token) % len(vector)
            vector[index] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def to_bytes(self, embedding: np.ndarray) -> bytes:
        return np.asarray(embedding, dtype=np.float32).tobytes()

    def from_bytes(self, data: bytes) -> np.ndarray:
        return np.frombuffer(data, dtype=np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def _make_graph(tmp_path: Path) -> MemoryGraph:
    return MemoryGraph(tmp_path / "health_test.db", FakeEmbeddingModel())


def _add(graph: MemoryGraph, label: str, content: str, node_type: NodeType = NodeType.FACT) -> str:
    result = graph.add_node(label=label, content=content, node_type=node_type)
    # add_node may return a Node directly or an AddNodeResult with .node
    if hasattr(result, "node"):
        return result.node.id
    return result.id


# ---------------------------------------------------------------------------
# graph_health — structural assertions
# ---------------------------------------------------------------------------

class TestGraphHealthEmpty:
    def test_empty_graph_returns_zeros(self, tmp_path):
        h = _make_graph(tmp_path).graph_health()
        assert h["total_nodes"] == 0
        assert h["total_edges"] == 0
        assert h["orphan_nodes"] == 0
        assert h["invalidated_nodes"] == 0
        assert h["health_score"]["overall"] == 100

    def test_empty_graph_age_distribution_zero(self, tmp_path):
        age = _make_graph(tmp_path).graph_health()["age_distribution"]
        assert age["updated_within_7d"] == 0
        assert age["updated_within_30d"] == 0
        assert age["stale_over_90d"] == 0


class TestGraphHealthWithNodes:
    def test_single_node_is_orphan(self, tmp_path):
        g = _make_graph(tmp_path)
        _add(g, "alpha", "unique alpha xyzzy content qrst")
        h = g.graph_health()
        assert h["total_nodes"] >= 1
        assert h["orphan_nodes"] >= 1
        assert h["total_edges"] == 0

    def test_node_type_counts_include_expected_types(self, tmp_path):
        g = _make_graph(tmp_path)
        # Use distinct content so dedup doesn't merge them
        _add(g, "fact-a", "zephyr alpha node unique xyzzy", NodeType.FACT)
        _add(g, "fact-b", "quorum beta node unique plugh", NodeType.FACT)
        _add(g, "dec-a",  "decided postgres xyzzy quorum plugh", NodeType.DECISION)
        h = g.graph_health()
        # At least one of each type must be present
        assert h["nodes_by_type"].get("fact", 0) >= 1
        assert h["nodes_by_type"].get("decision", 0) >= 1

    def test_edges_reduce_orphan_count(self, tmp_path):
        g = _make_graph(tmp_path)
        n1 = _add(g, "src-node", "unique src xyzzy content alpha")
        n2 = _add(g, "tgt-node", "unique tgt plugh content beta")
        g.add_edge(source_id=n1, target_id=n2, relationship=RelationType.RELATES_TO)
        h = g.graph_health()
        assert h["total_edges"] >= 1
        assert h["orphan_nodes"] == 0

    def test_edges_by_type_counted(self, tmp_path):
        g = _make_graph(tmp_path)
        n1 = _add(g, "src2", "unique source alpha xyzzy zeta")
        n2 = _add(g, "tgt2", "unique target beta plugh theta")
        g.add_edge(source_id=n1, target_id=n2, relationship=RelationType.RELATES_TO)
        h = g.graph_health()
        assert h["edges_by_type"].get("relates_to", 0) >= 1

    def test_invalidated_nodes_counted(self, tmp_path):
        g = _make_graph(tmp_path)
        nid = _add(g, "expiring", "unique expiring node xyzzy zeta")
        past = datetime.now(timezone.utc) - timedelta(days=1)
        g.update_node(node_id=nid, valid_to=past)
        h = g.graph_health()
        assert h["invalidated_nodes"] >= 1

    def test_recently_added_nodes_appear_in_7d_bucket(self, tmp_path):
        g = _make_graph(tmp_path)
        _add(g, "fresh", "freshly added unique xyzzy node alpha beta")
        age = g.graph_health()["age_distribution"]
        assert age["updated_within_7d"] >= 1
        assert age["updated_within_30d"] >= 1


# ---------------------------------------------------------------------------
# _compute_health_score — pure-function unit tests (no graph needed)
# ---------------------------------------------------------------------------

class TestComputeHealthScore:
    def test_zero_nodes_returns_perfect(self):
        score = MemoryGraph._compute_health_score(
            total_nodes=0, orphan_count=0, invalidated_count=0, stale_count=0
        )
        assert score == {"overall": 100, "connectivity": 100, "freshness": 100, "validity": 100}

    def test_all_orphans_tanks_connectivity(self):
        score = MemoryGraph._compute_health_score(
            total_nodes=10, orphan_count=10, invalidated_count=0, stale_count=0
        )
        assert score["connectivity"] == 0
        assert score["overall"] <= 60  # freshness + validity still contribute

    def test_all_stale_tanks_freshness(self):
        score = MemoryGraph._compute_health_score(
            total_nodes=10, orphan_count=0, invalidated_count=0, stale_count=10
        )
        assert score["freshness"] == 0

    def test_all_invalidated_tanks_validity(self):
        score = MemoryGraph._compute_health_score(
            total_nodes=10, orphan_count=0, invalidated_count=10, stale_count=0
        )
        assert score["validity"] == 0

    def test_healthy_graph_scores_high(self):
        score = MemoryGraph._compute_health_score(
            total_nodes=100, orphan_count=2, invalidated_count=1, stale_count=5
        )
        assert score["overall"] >= 80

    def test_scores_capped_at_100(self):
        score = MemoryGraph._compute_health_score(
            total_nodes=100, orphan_count=0, invalidated_count=0, stale_count=0
        )
        for key, val in score.items():
            assert val <= 100, f"{key} exceeds 100: {val}"

    def test_dimensions_are_non_negative(self):
        score = MemoryGraph._compute_health_score(
            total_nodes=5, orphan_count=5, invalidated_count=5, stale_count=5
        )
        for key, val in score.items():
            assert val >= 0, f"{key} should be >= 0, got {val}"
