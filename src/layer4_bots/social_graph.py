"""
TransitMind Sogamoso — Layer 4: Social Graph Module
======================================================
Graph G=(V,E,W) of bot users for prioritized alert dissemination.
Implements betweenness centrality + k-shell decomposition + SIR simulation.

Paper reference: Sections III and IV —
"Grafos de Redes Sociales para la Priorización Estratégica de Alertas Ciudadanas"
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from src.shared.logger import get_logger
from src.shared.utils import get_project_root

logger = get_logger("layer4.social_graph")

# Try importing EoN; fallback to analytical estimation if unavailable
try:
    import EoN
    _EON_AVAILABLE = True
except ImportError:
    _EON_AVAILABLE = False
    logger.warning("eon_not_available", msg="Using analytical SIR fallback")

# Try importing community detection (python-louvain)
try:
    import community as community_louvain
    _LOUVAIN_AVAILABLE = True
except ImportError:
    _LOUVAIN_AVAILABLE = False


class SocialGraphModule:
    """
    Social graph for prioritized alert dissemination.
    score(v) = α·BC_norm(v) + (1-α)·ks_norm(v)
    """

    def __init__(self, config: dict):
        sg_cfg = config.get("social_graph", {})
        self._enabled = sg_cfg.get("enabled", True)
        self._w1 = sg_cfg.get("w1_corridor", 0.5)
        self._w2 = sg_cfg.get("w2_temporal", 0.3)
        self._w3 = sg_cfg.get("w3_coconsult", 0.2)
        self._theta = sg_cfg.get("edge_threshold", 0.30)
        self._alpha = sg_cfg.get("alpha", 0.6)
        self._k = sg_cfg.get("first_wave_k", 5)
        self._broadcast_delay = sg_cfg.get("broadcast_delay_minutes", 2)
        self._n_simulations = sg_cfg.get("sir_simulations", 500)
        self._beta = sg_cfg.get("sir_beta", 0.3)
        self._gamma = sg_cfg.get("sir_gamma", 0.1)
        self._coverage_target = sg_cfg.get("coverage_target", 0.80)
        self._seed_file = sg_cfg.get("seed_nodes_file", "")
        self._edge_update_hours = sg_cfg.get("edge_update_hours", 24)
        self._centrality_update_hours = sg_cfg.get("centrality_update_hours", 6)

        # Main graph (NetworkX, undirected weighted)
        self._graph: nx.Graph = nx.Graph()
        # Cache: {user_id: score}
        self._propagator_ranking: Dict[str, float] = {}
        # Update timestamps
        self._last_edge_update: float = 0.0
        self._last_centrality_update: float = 0.0

        # Co-consult tracking: list of (user_id, intersection_id, timestamp)
        self._recent_queries: List[Dict[str, Any]] = []
        # Co-consult counts: {(u,v): count}
        self._coconsult_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self._max_coconsult: int = 1

        # Initialize with seed nodes
        self._load_seed_nodes()

    # ---- Graph Construction ----

    def _load_seed_nodes(self):
        """Load seed nodes from JSON or create 5 synthetic ones."""
        seed_path = None
        if self._seed_file:
            seed_path = get_project_root() / self._seed_file

        nodes = []
        if seed_path and seed_path.exists():
            try:
                with open(seed_path, "r", encoding="utf-8") as f:
                    nodes = json.load(f)
                logger.info("seed_nodes_loaded", count=len(nodes), path=str(seed_path))
            except Exception as e:
                logger.warning("seed_nodes_load_failed", error=str(e))

        if not nodes:
            # Create 5 synthetic seed nodes for cold start
            nodes = self._create_default_seeds()
            logger.info("seed_nodes_created_default", count=len(nodes))

        for node in nodes:
            self.register_user(node)

        # Compute initial edges and centrality
        if self._graph.number_of_nodes() > 1:
            self.rebuild_edges()
            self.compute_centrality()

    def _create_default_seeds(self) -> List[dict]:
        """Create 5 default seed nodes for cold start."""
        now = datetime.now(timezone.utc).isoformat()
        corridors_map = {
            "seed_001": (["av_castellana_entrada", "av_castellana_salida"], "despachador", [5,6,7,8,16,17,18]),
            "seed_002": (["carrera_11_norte", "carrera_11_sur", "calle_14_centro_historico"], "despachador", [6,7,8,9,17,18,19]),
            "seed_003": (["carrera_11_norte", "av_castellana_entrada"], "conductor", [6,7,8,17,18]),
            "seed_004": (["acceso_morca", "calle_14_centro_historico"], "conductor", [5,6,7,12,13,17,18]),
            "seed_005": (["carrera_11_sur", "calle_14_centro_historico", "carrera_11_norte"], "lider_barrio", [7,8,12,17,18,19]),
        }
        seeds = []
        for uid, (corrs, role, hours) in corridors_map.items():
            seeds.append({
                "user_id": uid, "role": role, "corridors": corrs,
                "peak_hours": hours, "query_count": 0, "is_seed": True,
                "registered_at": now, "last_active": now,
            })
        return seeds

    def register_user(self, user_node: dict):
        """Add or update a user node in the graph."""
        uid = user_node.get("user_id", "")
        if not uid:
            return

        now = datetime.now(timezone.utc).isoformat()
        if self._graph.has_node(uid):
            # Update existing
            attrs = self._graph.nodes[uid]
            attrs["last_active"] = user_node.get("last_active", now)
            attrs["query_count"] = attrs.get("query_count", 0) + user_node.get("query_count", 0)
            # Merge corridors
            existing_corrs = set(attrs.get("corridors", []))
            new_corrs = set(user_node.get("corridors", []))
            attrs["corridors"] = list(existing_corrs | new_corrs)
            # Merge peak_hours
            existing_hours = set(attrs.get("peak_hours", []))
            new_hours = set(user_node.get("peak_hours", []))
            attrs["peak_hours"] = sorted(list(existing_hours | new_hours))
        else:
            self._graph.add_node(uid, **{
                "corridors": user_node.get("corridors", []),
                "peak_hours": user_node.get("peak_hours", []),
                "query_count": user_node.get("query_count", 0),
                "is_seed": user_node.get("is_seed", False),
                "role": user_node.get("role", "vecino"),
                "registered_at": user_node.get("registered_at", now),
                "last_active": user_node.get("last_active", now),
            })

            # Compute edges incrementally with all existing nodes
            for other_uid in list(self._graph.nodes):
                if other_uid != uid:
                    w = self._compute_edge_weight(uid, other_uid)
                    if w > 0:
                        self._graph.add_edge(uid, other_uid, weight=w)

    def register_query(self, query: dict):
        """Register a bot query to build/update edges."""
        uid = query.get("user_id", "")
        iid = query.get("intersection_id", "")
        ts_str = query.get("timestamp", datetime.now(timezone.utc).isoformat())

        if not uid or not iid:
            return

        # Update node attributes
        if self._graph.has_node(uid):
            attrs = self._graph.nodes[uid]
            attrs["query_count"] = attrs.get("query_count", 0) + 1
            attrs["last_active"] = ts_str
            corrs = set(attrs.get("corridors", []))
            corrs.add(iid)
            attrs["corridors"] = list(corrs)

        # Track for co-consult
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            ts = datetime.now(timezone.utc)

        self._recent_queries.append({"user_id": uid, "intersection_id": iid, "timestamp": ts})

        # Check co-consult: same intersection within 10 min window
        window_seconds = 600  # 10 minutes
        for rq in self._recent_queries[-200:]:  # limit lookback
            if rq["user_id"] != uid and rq["intersection_id"] == iid:
                try:
                    delta = abs((ts - rq["timestamp"]).total_seconds())
                except (TypeError, AttributeError):
                    continue
                if delta <= window_seconds:
                    pair = tuple(sorted([uid, rq["user_id"]]))
                    self._coconsult_counts[pair] += 1
                    self._max_coconsult = max(self._max_coconsult, self._coconsult_counts[pair])

        # Prune old queries (keep last 1000)
        if len(self._recent_queries) > 1000:
            self._recent_queries = self._recent_queries[-500:]

    # ---- Edge Weight Computation ----

    def _sim_corridor(self, u: dict, v: dict) -> float:
        """Jaccard similarity on corridor sets."""
        u_corrs = set(u.get("corridors", []))
        v_corrs = set(v.get("corridors", []))
        if not u_corrs and not v_corrs:
            return 0.0
        union = u_corrs | v_corrs
        if not union:
            return 0.0
        return len(u_corrs & v_corrs) / len(union)

    def _sim_temporal(self, u: dict, v: dict) -> float:
        """Cosine similarity on 24-dimensional hourly vectors."""
        u_hours = u.get("peak_hours", [])
        v_hours = v.get("peak_hours", [])
        if not u_hours and not v_hours:
            return 0.0

        # Build 24-dim vectors
        u_vec = [0.0] * 24
        v_vec = [0.0] * 24
        for h in u_hours:
            if 0 <= h < 24:
                u_vec[h] = 1.0
        for h in v_hours:
            if 0 <= h < 24:
                v_vec[h] = 1.0

        # Cosine similarity
        dot = sum(a * b for a, b in zip(u_vec, v_vec))
        mag_u = math.sqrt(sum(a * a for a in u_vec))
        mag_v = math.sqrt(sum(b * b for b in v_vec))
        if mag_u == 0 or mag_v == 0:
            return 0.0
        return dot / (mag_u * mag_v)

    def _sim_coconsult(self, u_id: str, v_id: str) -> float:
        """Co-consult count normalized by max observed."""
        pair = tuple(sorted([u_id, v_id]))
        count = self._coconsult_counts.get(pair, 0)
        if self._max_coconsult <= 0:
            return 0.0
        return count / self._max_coconsult

    def _compute_edge_weight(self, u_id: str, v_id: str) -> float:
        """W(u,v) = w1·sim_corridor + w2·sim_temporal + w3·co_consult. Returns 0.0 if < threshold."""
        u_data = self._graph.nodes.get(u_id, {})
        v_data = self._graph.nodes.get(v_id, {})

        s1 = self._sim_corridor(u_data, v_data)
        s2 = self._sim_temporal(u_data, v_data)
        s3 = self._sim_coconsult(u_id, v_id)

        w = self._w1 * s1 + self._w2 * s2 + self._w3 * s3
        return w if w >= self._theta else 0.0

    def rebuild_edges(self):
        """Rebuild ALL edges from scratch. O(|V|²). Call every 24h."""
        nodes = list(self._graph.nodes)
        # Remove existing edges
        self._graph.remove_edges_from(list(self._graph.edges))

        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                w = self._compute_edge_weight(u, v)
                if w > 0:
                    self._graph.add_edge(u, v, weight=w)

        self._last_edge_update = time.time()
        logger.info("edges_rebuilt", nodes=len(nodes), edges=self._graph.number_of_edges())

    # ---- Centrality & Ranking ----

    def compute_centrality(self):
        """Compute betweenness + k-shell + combined score."""
        n = self._graph.number_of_nodes()
        if n < 2:
            self._propagator_ranking = {uid: 0.5 for uid in self._graph.nodes}
            self._last_centrality_update = time.time()
            return

        # Betweenness centrality (Brandes)
        bc = nx.betweenness_centrality(self._graph, weight="weight", normalized=True)

        # K-shell decomposition (coreness)
        ks = nx.core_number(self._graph)
        max_ks = max(ks.values()) if ks else 1
        if max_ks == 0:
            max_ks = 1

        # Combined score
        self._propagator_ranking = {}
        for uid in self._graph.nodes:
            bc_norm = bc.get(uid, 0.0)
            ks_norm = ks.get(uid, 0) / max_ks
            score = self._alpha * bc_norm + (1 - self._alpha) * ks_norm
            self._propagator_ranking[uid] = round(score, 6)

        self._last_centrality_update = time.time()

        # Persist rankings
        self._save_rankings()

        logger.info(
            "centrality_computed", nodes=n, edges=self._graph.number_of_edges(),
            top_score=max(self._propagator_ranking.values()) if self._propagator_ranking else 0,
        )

    def _save_rankings(self):
        """Persist propagator rankings to disk."""
        try:
            out_dir = get_project_root() / "data" / "layer4_outputs" / "social_graph" / "propagator_rankings"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"ranking_{ts}.json"

            ranking_list = []
            for uid, score in sorted(self._propagator_ranking.items(), key=lambda x: x[1], reverse=True):
                attrs = self._graph.nodes.get(uid, {})
                ranking_list.append({
                    "user_id": uid, "score": score,
                    "role": attrs.get("role", "vecino"),
                    "corridors": attrs.get("corridors", []),
                })

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(ranking_list, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning("save_rankings_failed", error=str(e))

    def get_propagators_for_corridor(self, corridor_intersections: List[str], k: int = None) -> List[str]:
        """
        Step 1: Filter subgraph Gc of users whose corridors overlap.
        Step 2: Apply ranking on Gc.
        Step 3: Simulate SIR; if coverage < target, expand k.
        Returns ordered list of user_ids (first-wave propagators).
        """
        if k is None:
            k = self._k

        # Step 1: filter subgraph
        corridor_set = set(corridor_intersections)
        gc_nodes = [
            uid for uid in self._graph.nodes
            if corridor_set & set(self._graph.nodes[uid].get("corridors", []))
        ]

        if not gc_nodes:
            return []

        # Step 2: rank by propagator score
        ranked = sorted(gc_nodes, key=lambda u: self._propagator_ranking.get(u, 0), reverse=True)
        top_k = ranked[:k]

        # Step 3: simulate SIR coverage
        if len(gc_nodes) > 1:
            subgraph = self._graph.subgraph(gc_nodes).copy()
            coverage = self.simulate_sir_coverage(top_k, subgraph)

            # If coverage < target and we have more nodes, expand k
            attempts = 0
            while coverage < self._coverage_target and len(top_k) < len(ranked) and attempts < 3:
                k_expanded = min(len(top_k) + 3, len(ranked))
                top_k = ranked[:k_expanded]
                coverage = self.simulate_sir_coverage(top_k, subgraph)
                attempts += 1
        else:
            coverage = 1.0

        return top_k

    def simulate_sir_coverage(self, seed_nodes: List[str], subgraph: nx.Graph = None) -> float:
        """Simulate SIR propagation. Returns fraction of nodes reached (0-1)."""
        graph = subgraph if subgraph is not None else self._graph
        n = graph.number_of_nodes()

        if n == 0:
            return 0.0
        if n <= len(seed_nodes):
            return 1.0

        if _EON_AVAILABLE and graph.number_of_edges() > 0:
            return self._sir_eon(seed_nodes, graph)
        else:
            return self._sir_analytical(seed_nodes, graph)

    def _sir_eon(self, seed_nodes: List[str], graph: nx.Graph) -> float:
        """Monte Carlo SIR simulation using EoN library."""
        try:
            n = graph.number_of_nodes()
            total_coverage = 0.0
            n_sims = min(self._n_simulations, 100)  # Cap for performance

            for _ in range(n_sims):
                initial_infecteds = seed_nodes
                t, S, I, R = EoN.fast_SIR(
                    graph, tau=self._beta, gamma=self._gamma,
                    initial_infecteds=initial_infecteds, tmax=5,
                )
                # Coverage = fraction recovered at end
                final_R = R[-1] if len(R) > 0 else 0
                total_coverage += final_R / n

            avg_coverage = total_coverage / n_sims
            return min(avg_coverage, 1.0)
        except Exception as e:
            logger.warning("eon_simulation_failed", error=str(e))
            return self._sir_analytical(seed_nodes, graph)

    def _sir_analytical(self, seed_nodes: List[str], graph: nx.Graph) -> float:
        """Analytical SIR estimation based on average degree."""
        n = graph.number_of_nodes()
        if n == 0:
            return 0.0

        k_seeds = len(seed_nodes)
        if k_seeds >= n:
            return 1.0

        # Estimate using branching factor
        degrees = dict(graph.degree())
        if not degrees:
            return k_seeds / n

        avg_degree = sum(degrees.values()) / len(degrees)
        R0 = self._beta * avg_degree / self._gamma

        if R0 <= 1:
            return k_seeds / n

        # Estimated final size: 1 - exp(-R0 * final_size)
        # Approximate: final_size ≈ 1 - 1/R0 (for large R0)
        estimated_reach = min(1.0, 1.0 - 1.0 / R0)
        # Scale by seed fraction
        coverage = min(1.0, k_seeds / n + estimated_reach * (1 - k_seeds / n))
        return coverage

    # ---- Public API ----

    def get_alert_order(self, affected_intersections: List[str], all_users: List[str]) -> dict:
        """
        Entry point for AlertEngine.
        Returns first_wave, broadcast, delay, coverage, and graph stats.
        """
        self.maybe_update()

        first_wave = self.get_propagators_for_corridor(affected_intersections, self._k)

        # Broadcast = all active users not in first_wave
        first_wave_set = set(first_wave)
        broadcast = [u for u in all_users if u not in first_wave_set]

        # Coverage estimate
        if self._graph.number_of_nodes() > 1:
            coverage = self.simulate_sir_coverage(first_wave)
        else:
            coverage = 1.0 if first_wave else 0.0

        n = self._graph.number_of_nodes()
        e = self._graph.number_of_edges()
        avg_deg = (2 * e / n) if n > 0 else 0.0

        return {
            "first_wave": first_wave,
            "broadcast": broadcast,
            "broadcast_delay_minutes": self._broadcast_delay,
            "expected_coverage_pct": round(coverage * 100, 1),
            "propagators_used": len(first_wave),
            "graph_stats": {
                "total_nodes": n,
                "active_nodes": n,  # simplified
                "edges": e,
                "avg_degree": round(avg_deg, 2),
            },
        }

    def get_graph_summary(self) -> dict:
        """Summary of current graph state for the dashboard."""
        n = self._graph.number_of_nodes()
        e = self._graph.number_of_edges()

        # Top propagators
        top = sorted(self._propagator_ranking.items(), key=lambda x: x[1], reverse=True)[:5]
        top_list = []
        for uid, score in top:
            attrs = self._graph.nodes.get(uid, {})
            top_list.append({
                "user_id": uid, "role": attrs.get("role", "vecino"),
                "score": score, "corridors": attrs.get("corridors", []),
            })

        # Community detection
        n_communities = 0
        if _LOUVAIN_AVAILABLE and n > 1 and e > 0:
            try:
                partition = community_louvain.best_partition(self._graph)
                n_communities = len(set(partition.values()))
            except Exception:
                pass

        last_update = ""
        if self._last_centrality_update > 0:
            last_update = datetime.fromtimestamp(self._last_centrality_update).isoformat()

        return {
            "total_users": n,
            "active_last_24h": n,  # simplified
            "edges": e,
            "top_propagators": top_list,
            "communities_detected": n_communities,
            "last_centrality_update": last_update,
        }

    def maybe_update(self):
        """Check if edges (24h) or centrality (6h) need recalculation."""
        now = time.time()
        if now - self._last_edge_update > self._edge_update_hours * 3600:
            self.rebuild_edges()
        if now - self._last_centrality_update > self._centrality_update_hours * 3600:
            self.compute_centrality()

    def get_all_user_ids(self) -> List[str]:
        """Return all user IDs in the graph."""
        return list(self._graph.nodes)
