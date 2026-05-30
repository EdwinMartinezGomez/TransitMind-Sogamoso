"""
Unit tests for Layer 4 SocialGraphModule.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.shared.utils import load_yaml_config
from src.layer4_bots.social_graph import SocialGraphModule


@pytest.fixture
def config():
    return load_yaml_config("layer4_config.yaml")


@pytest.fixture
def graph(config):
    return SocialGraphModule(config)


def _create_test_users(graph: SocialGraphModule, n: int = 20):
    """Register n synthetic users in the graph."""
    corridors_pool = [
        ["carrera_11_norte", "carrera_11_sur"],
        ["av_castellana_entrada", "av_castellana_salida"],
        ["calle_14_centro_historico", "acceso_morca"],
        ["carrera_11_norte", "av_castellana_entrada"],
        ["carrera_11_sur", "calle_14_centro_historico"],
    ]
    roles = ["vecino", "conductor", "lider_barrio", "vecino", "conductor"]

    for i in range(n):
        graph.register_user({
            "user_id": f"test_user_{i:03d}",
            "corridors": corridors_pool[i % len(corridors_pool)],
            "peak_hours": [7, 8, 17, 18] if i % 2 == 0 else [6, 7, 12, 13],
            "query_count": i * 3,
            "is_seed": False,
            "role": roles[i % len(roles)],
        })


class TestSocialGraph:
    """Tests for SocialGraphModule."""

    def test_seed_nodes_loaded(self, graph):
        """Graph should have seed nodes on init."""
        assert graph._graph.number_of_nodes() >= 5, "Should have at least 5 seed nodes"

    def test_register_user(self, graph):
        """Registering a user should add a node."""
        initial = graph._graph.number_of_nodes()
        graph.register_user({
            "user_id": "new_user_001",
            "corridors": ["carrera_11_norte"],
            "peak_hours": [7, 8],
            "query_count": 0,
            "is_seed": False,
            "role": "vecino",
        })
        assert graph._graph.number_of_nodes() == initial + 1

    def test_register_user_update(self, graph):
        """Re-registering a user should update, not duplicate."""
        graph.register_user({"user_id": "upd_001", "corridors": ["carrera_11_norte"], "peak_hours": [7]})
        n1 = graph._graph.number_of_nodes()
        graph.register_user({"user_id": "upd_001", "corridors": ["carrera_11_sur"], "peak_hours": [8]})
        n2 = graph._graph.number_of_nodes()
        assert n1 == n2, "Should not duplicate node"
        # Corridors should merge
        corrs = set(graph._graph.nodes["upd_001"]["corridors"])
        assert "carrera_11_norte" in corrs and "carrera_11_sur" in corrs

    def test_compute_centrality(self, graph):
        """compute_centrality should not raise and should populate rankings."""
        _create_test_users(graph, 20)
        graph.rebuild_edges()
        graph.compute_centrality()
        assert len(graph._propagator_ranking) > 0

    def test_scores_bounded(self, graph):
        """All scores should be in [0, 1]."""
        _create_test_users(graph, 20)
        graph.rebuild_edges()
        graph.compute_centrality()
        for uid, score in graph._propagator_ranking.items():
            assert 0.0 <= score <= 1.0, f"Score out of bounds: {uid}={score}"

    def test_propagators_for_corridor(self, graph):
        """get_propagators_for_corridor should return a list of user_ids."""
        _create_test_users(graph, 20)
        graph.rebuild_edges()
        graph.compute_centrality()
        propagators = graph.get_propagators_for_corridor(["carrera_11_norte"])
        assert isinstance(propagators, list)
        assert len(propagators) > 0

    def test_sir_coverage_bounded(self, graph):
        """SIR coverage should be in [0, 1]."""
        _create_test_users(graph, 15)
        graph.rebuild_edges()
        graph.compute_centrality()
        seeds = list(graph._graph.nodes)[:3]
        coverage = graph.simulate_sir_coverage(seeds)
        assert 0.0 <= coverage <= 1.0, f"Coverage out of bounds: {coverage}"

    def test_get_alert_order(self, graph):
        """get_alert_order should return proper structure."""
        _create_test_users(graph, 15)
        graph.rebuild_edges()
        graph.compute_centrality()
        all_users = graph.get_all_user_ids()
        order = graph.get_alert_order(["carrera_11_norte"], all_users)
        assert "first_wave" in order
        assert "broadcast" in order
        assert "expected_coverage_pct" in order
        assert "graph_stats" in order
        assert order["expected_coverage_pct"] >= 0

    def test_graph_summary(self, graph):
        """get_graph_summary should return dict with expected keys."""
        summary = graph.get_graph_summary()
        assert "total_users" in summary
        assert "edges" in summary
        assert "top_propagators" in summary
        assert summary["total_users"] >= 5

    def test_sim_corridor_jaccard(self, graph):
        """Corridor similarity should work correctly."""
        u = {"corridors": ["a", "b", "c"]}
        v = {"corridors": ["b", "c", "d"]}
        sim = graph._sim_corridor(u, v)
        # Jaccard: |{b,c}| / |{a,b,c,d}| = 2/4 = 0.5
        assert abs(sim - 0.5) < 0.01

    def test_sim_temporal_cosine(self, graph):
        """Temporal similarity should work for identical vectors."""
        u = {"peak_hours": [7, 8, 17, 18]}
        v = {"peak_hours": [7, 8, 17, 18]}
        sim = graph._sim_temporal(u, v)
        assert abs(sim - 1.0) < 0.01, "Identical peak hours should have sim=1.0"
