"""Tests for waggle/memory_lifecycle.py

Covers:
- compute_importance decay math (known inputs → known outputs)
- base importance by node type
- edge reinforcement slows decay
- access frequency boost
- identify_prunable age gate and threshold
- active validity window protection
"""
from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta

import pytest

from waggle.memory_lifecycle import compute_importance, identify_prunable
from waggle.models import Node, NodeType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node(
    node_type: NodeType = NodeType.FACT,
    *,
    access_count: int = 0,
    days_since_access: float = 0.0,
    days_old: float = 0.0,
    valid_to: datetime | None = None,
) -> Node:
    now = datetime.now(timezone.utc)
    last_accessed = now - timedelta(days=days_since_access)
    created = now - timedelta(days=days_old)
    n = Node(
        label="test node",
        content="test content",
        node_type=node_type,
        created_at=created,
        updated_at=last_accessed,
        access_count=access_count,
        valid_to=valid_to,
    )
    # Manually set last_accessed_at so decay uses it
    object.__setattr__(n, "last_accessed_at", last_accessed)
    return n


# ---------------------------------------------------------------------------
# compute_importance — decay math
# ---------------------------------------------------------------------------

class TestComputeImportanceDecay:
    def test_fresh_node_scores_near_base(self):
        """A just-created fact node with no access should score close to its base (0.75)."""
        node = _node(NodeType.FACT, days_since_access=0)
        score = compute_importance(node, half_life_days=14.0)
        assert score == pytest.approx(0.75, abs=0.05)

    def test_score_at_one_half_life(self):
        """After one half-life, score should be roughly half the base."""
        node = _node(NodeType.FACT, days_since_access=14.0)
        score = compute_importance(node, half_life_days=14.0)
        # base=0.75, decay=0.5 → 0.375 + small access_boost=0
        assert score == pytest.approx(0.75 * 0.5, abs=0.05)

    def test_score_at_two_half_lives(self):
        """After two half-lives the score should be roughly 25% of base."""
        node = _node(NodeType.FACT, days_since_access=28.0)
        score = compute_importance(node, half_life_days=14.0)
        assert score == pytest.approx(0.75 * 0.25, abs=0.05)

    def test_fresh_node_always_above_zero(self):
        node = _node(NodeType.NOTE, days_since_access=0)
        assert compute_importance(node) > 0.0

    def test_very_old_node_approaches_zero(self):
        """A node untouched for 10× the half-life should score very low."""
        node = _node(NodeType.NOTE, days_since_access=140.0)
        score = compute_importance(node, half_life_days=14.0)
        assert score < 0.05

    def test_output_clamped_to_unit_interval(self):
        node = _node(NodeType.DECISION, access_count=1000)
        score = compute_importance(node)
        assert 0.0 <= score <= 1.0

    def test_negative_half_life_uses_default(self):
        """Invalid half-life should not crash; falls back to 14.0."""
        node = _node(NodeType.FACT, days_since_access=14.0)
        score_safe = compute_importance(node, half_life_days=-1.0)
        score_ref = compute_importance(node, half_life_days=14.0)
        assert score_safe == pytest.approx(score_ref, abs=0.001)


# ---------------------------------------------------------------------------
# compute_importance — base importance by node type
# ---------------------------------------------------------------------------

class TestBaseImportanceByType:
    @pytest.mark.parametrize("node_type,expected_min", [
        (NodeType.DECISION, 0.90),
        (NodeType.FACT,     0.65),
        (NodeType.NOTE,     0.50),
    ])
    def test_type_ordering(self, node_type, expected_min):
        """Decisions should score higher than facts, facts higher than notes."""
        node = _node(node_type, days_since_access=0)
        score = compute_importance(node, half_life_days=14.0)
        assert score >= expected_min, f"{node_type} scored {score} < {expected_min}"

    def test_decision_beats_note(self):
        decision = _node(NodeType.DECISION, days_since_access=7.0)
        note = _node(NodeType.NOTE, days_since_access=0.0)
        assert compute_importance(decision) > compute_importance(note)


# ---------------------------------------------------------------------------
# compute_importance — edge reinforcement
# ---------------------------------------------------------------------------

class TestEdgeReinforcement:
    def test_more_edges_slow_decay(self):
        """A node with many edges should decay slower than an isolated one."""
        isolated = _node(NodeType.FACT, days_since_access=14.0)
        connected = _node(NodeType.FACT, days_since_access=14.0)
        score_isolated = compute_importance(isolated, half_life_days=14.0, edge_count=0)
        score_connected = compute_importance(connected, half_life_days=14.0, edge_count=20)
        assert score_connected > score_isolated

    def test_edge_reinforcement_cap(self):
        """Even with infinite edges, decay should not be eliminated entirely."""
        node = _node(NodeType.FACT, days_since_access=200.0)
        score = compute_importance(node, half_life_days=14.0, edge_count=10_000)
        # 200 days at 4× half-life (56d) → 0.5^(200/56) ≈ 0.08 × 0.75 ≈ 0.06
        assert score < 0.5, "Edge reinforcement must not eliminate decay for very old nodes"

    def test_zero_edges_matches_no_edge_count_arg(self):
        node = _node(NodeType.FACT, days_since_access=7.0)
        s1 = compute_importance(node, half_life_days=14.0, edge_count=0)
        s2 = compute_importance(node, half_life_days=14.0)
        assert s1 == pytest.approx(s2, abs=1e-9)


# ---------------------------------------------------------------------------
# compute_importance — access frequency boost
# ---------------------------------------------------------------------------

class TestAccessFrequencyBoost:
    def test_frequent_access_boosts_score(self):
        cold = _node(NodeType.FACT, access_count=0,  days_since_access=7.0)
        hot  = _node(NodeType.FACT, access_count=50, days_since_access=7.0)
        assert compute_importance(hot) > compute_importance(cold)

    def test_access_boost_is_capped(self):
        """Access boost must not push score above 1.0."""
        node = _node(NodeType.DECISION, access_count=10_000, days_since_access=0)
        assert compute_importance(node) <= 1.0


# ---------------------------------------------------------------------------
# identify_prunable
# ---------------------------------------------------------------------------

class TestIdentifyPrunable:
    def test_fresh_nodes_never_prunable(self):
        """Nodes younger than min_age_days must never appear in output."""
        nodes = [_node(NodeType.NOTE, days_since_access=200, days_old=30)]
        result = identify_prunable(nodes, min_age_days=90, importance_threshold=0.5)
        assert result == []

    def test_stale_low_score_node_is_prunable(self):
        """Very old, untouched note with no edges should be prunable."""
        node = _node(NodeType.NOTE, days_since_access=200, days_old=200)
        result = identify_prunable(
            [node],
            half_life_days=14.0,
            importance_threshold=0.05,
            min_age_days=90,
        )
        assert node.id in result

    def test_high_importance_node_not_prunable(self):
        """A frequently accessed decision node should not be prunable."""
        node = _node(NodeType.DECISION, access_count=30, days_since_access=2, days_old=200)
        result = identify_prunable(
            [node],
            half_life_days=14.0,
            importance_threshold=0.05,
            min_age_days=90,
        )
        assert node.id not in result

    def test_active_validity_window_protects_node(self):
        """A node with valid_to in the future must not be pruned."""
        future = datetime.now(timezone.utc) + timedelta(days=30)
        node = _node(NodeType.NOTE, days_since_access=200, days_old=200, valid_to=future)
        result = identify_prunable(
            [node],
            half_life_days=14.0,
            importance_threshold=0.5,
            min_age_days=90,
        )
        assert node.id not in result

    def test_expired_validity_window_does_not_protect(self):
        """A node whose valid_to is in the past is still eligible for pruning."""
        past = datetime.now(timezone.utc) - timedelta(days=1)
        node = _node(NodeType.NOTE, days_since_access=200, days_old=200, valid_to=past)
        result = identify_prunable(
            [node],
            half_life_days=14.0,
            importance_threshold=0.05,
            min_age_days=90,
        )
        assert node.id in result

    def test_empty_graph_returns_empty(self):
        assert identify_prunable([]) == []

    def test_returns_ids_not_nodes(self):
        node = _node(NodeType.NOTE, days_since_access=200, days_old=200)
        result = identify_prunable([node], half_life_days=14.0, importance_threshold=0.5, min_age_days=90)
        if result:
            assert isinstance(result[0], str)
