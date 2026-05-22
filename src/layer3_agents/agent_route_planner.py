"""
TransitMind Sogamoso — Agent Route Planner
============================================
Agent 5: Consolidates and prioritizes alternative routes
suggested by Layer 2. Filters out routes leading to already
congested intersections and provides fallback routes.
"""

from typing import Any, Dict, List

from src.shared.logger import get_logger

logger = get_logger("layer3.agent_route_planner")

# Known route-to-intersection mapping for Sogamoso
ROUTE_TO_INTERSECTION = {
    "Carrera 11": ["carrera_11_norte", "carrera_11_sur"],
    "Calle 14": ["calle_14_centro_historico"],
    "Variante Morca": ["acceso_morca"],
    "Avenida Castellana": ["av_castellana_entrada", "av_castellana_salida"],
    # "Avenida Industrial" is NOT a pilot intersection — always available
}

SEVERITY_ORDER = {"baja": 0, "media": 1, "alta": 2, "critica": 3}

FALLBACK_ROUTE = "Avenida Industrial"


class RoutePlannerAgent:
    """Consolidates and prioritizes alternative routes from causal analyses."""

    def __init__(self, config: dict):
        self._max_routes = config["agents"]["route_planner"]["max_alternative_routes"]
        self._min_severity = config["agents"]["route_planner"]["min_severity_for_reroute"]

    def _severity_meets_threshold(self, severity: str) -> bool:
        """True if severity >= min_severity_for_reroute."""
        return SEVERITY_ORDER.get(severity, 0) >= SEVERITY_ORDER.get(self._min_severity, 1)

    def _filter_congested_routes(
        self,
        routes: List[str],
        all_analyses: Dict[str, Any],
    ) -> List[str]:
        """
        Remove routes that correspond to intersections with
        severity "alta" or "critica".
        """
        congested_intersections = set()
        for iid, analysis in all_analyses.items():
            causal_ctx = analysis.get("causal_context", {})
            sev = causal_ctx.get("severity", "baja")
            if SEVERITY_ORDER.get(sev, 0) >= SEVERITY_ORDER.get("alta", 2):
                congested_intersections.add(iid)

        filtered = []
        for route in routes:
            # Check if route maps to a congested intersection
            mapped_intersections = ROUTE_TO_INTERSECTION.get(route, [])
            if not mapped_intersections:
                # Unknown route — keep it (e.g. "Avenida Industrial")
                filtered.append(route)
            elif not any(iid in congested_intersections for iid in mapped_intersections):
                filtered.append(route)
            else:
                logger.debug(
                    "route_filtered_congested",
                    route=route,
                    reason="intersection_congested",
                )

        return filtered

    def run(self, state: dict) -> dict:
        """
        Main logic:
        1. Read causal_analyses
        2. For each intersection:
           a. If severity < min_severity → no reroute needed
           b. If severity >= min_severity → take alternative routes,
              filter congested ones, limit to max_alternative_routes
        3. Detect reroute_needed
        """
        causal_analyses = state.get("causal_analyses", {})

        route_plans: Dict[str, List[str]] = {}
        reroute_needed = False

        for intersection_id, analysis in causal_analyses.items():
            causal_ctx = analysis.get("causal_context", {})
            severity = causal_ctx.get("severity", "baja")

            if not self._severity_meets_threshold(severity):
                route_plans[intersection_id] = []
                continue

            # Get alternative routes from recommendations
            recommendations = analysis.get("recommendations", {})
            alt_routes = list(recommendations.get("alternative_routes", []))

            # Filter out congested routes
            alt_routes = self._filter_congested_routes(alt_routes, causal_analyses)

            # Limit to max_alternative_routes
            alt_routes = alt_routes[: self._max_routes]

            # Fallback if empty
            if not alt_routes:
                alt_routes = [FALLBACK_ROUTE]

            route_plans[intersection_id] = alt_routes
            reroute_needed = True

        # Log summary
        routes_assigned = sum(1 for r in route_plans.values() if r)
        all_routes = [r for routes in route_plans.values() for r in routes]
        logger.info(
            "route_planner_complete",
            intersections_with_reroute=routes_assigned,
            reroute_needed=reroute_needed,
            total_routes=len(all_routes),
        )

        return {
            **state,
            "route_plans": route_plans,
            "reroute_needed": reroute_needed,
        }


def route_planner_node(state: dict) -> dict:
    """LangGraph node function for the Route Planner."""
    from src.shared.utils import load_yaml_config

    config = load_yaml_config("agents_config.yaml")
    agent = RoutePlannerAgent(config)
    return agent.run(state)
