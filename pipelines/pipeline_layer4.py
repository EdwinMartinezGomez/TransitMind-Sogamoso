"""
TransitMind Sogamoso — Layer 4 Validation Pipeline
=====================================================
Validates that Layer 4 meets acceptance criteria.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.shared.logger import get_logger, setup_logger

setup_logger("pipeline_layer4", level="INFO")
logger = get_logger("pipeline_layer4")


# ============================================
# Sample Decision Data for Testing
# ============================================

SAMPLE_DECISIONS = [
    {
        "intersection_id": "carrera_11_norte",
        "severity": "alta",
        "congestion_level": 0.78,
        "traffic_light_command": {
            "green_extension_seconds": 25,
            "priority_direction": "norte_sur",
            "cycle_adjustment_percent": 15,
        },
        "active_routes": ["Avenida Industrial", "Calle 14"],
        "citizen_alert": "Alerta movilidad: congestión alta en Carrera 11 Norte.",
        "agent_trace": ["sensor", "predictor", "gan_simulator"],
    },
    {
        "intersection_id": "av_castellana_entrada",
        "severity": "critica",
        "congestion_level": 0.92,
        "traffic_light_command": {
            "green_extension_seconds": 30,
            "priority_direction": "este_oeste",
            "cycle_adjustment_percent": 20,
        },
        "active_routes": ["Calle 14", "Acceso Morca"],
        "citizen_alert": "Alerta movilidad: congestión critica.",
        "agent_trace": ["sensor", "predictor"],
    },
    {
        "intersection_id": "calle_14_centro_historico",
        "severity": "media",
        "congestion_level": 0.55,
        "traffic_light_command": {
            "green_extension_seconds": 10,
            "priority_direction": "norte_sur",
            "cycle_adjustment_percent": 5,
        },
        "active_routes": ["Carrera 11 Sur"],
        "citizen_alert": "Alerta movilidad: congestión media.",
        "agent_trace": ["sensor"],
    },
    {
        "intersection_id": "carrera_11_sur",
        "severity": "media",
        "congestion_level": 0.48,
        "traffic_light_command": {
            "green_extension_seconds": 5,
            "priority_direction": "norte_sur",
            "cycle_adjustment_percent": 3,
        },
        "active_routes": [],
        "citizen_alert": "Tráfico moderado.",
        "agent_trace": ["sensor"],
    },
    {
        "intersection_id": "acceso_morca",
        "severity": "baja",
        "congestion_level": 0.25,
        "traffic_light_command": {
            "green_extension_seconds": 0,
            "priority_direction": "norte_sur",
            "cycle_adjustment_percent": 0,
        },
        "active_routes": [],
        "citizen_alert": "Tráfico normal.",
        "agent_trace": ["sensor"],
    },
    {
        "intersection_id": "av_castellana_salida",
        "severity": "alta",
        "congestion_level": 0.80,
        "traffic_light_command": {
            "green_extension_seconds": 20,
            "priority_direction": "este_oeste",
            "cycle_adjustment_percent": 12,
        },
        "active_routes": ["Calle 14", "Carrera 11 Norte"],
        "citizen_alert": "Alerta movilidad: congestión alta.",
        "agent_trace": ["sensor", "predictor"],
    },
]

SAMPLE_FINAL_DECISION = {
    "cycle_id": "test-cycle-001",
    "timestamp": "2024-06-15T07:23:00Z",
    "intersections_analyzed": 6,
    "decisions": SAMPLE_DECISIONS,
    "global_tmc_reduction_percent": 31.2,
    "monitor_report": {
        "anomalies_detected": 0,
        "hallucinations_blocked": 0,
        "cycle_duration_ms": 4200,
        "agents_healthy": 7,
    },
}

# Forbidden terms in citizen-facing output
FORBIDDEN_TERMS = [
    "congestion_level", "severity", "green_extension", "cycle_adjustment",
    "priority_direction", "agent_trace", "GAN", "LLM", "agente",
    "traffic_light_command", "intersection_id",
]


class Layer4Pipeline:
    """Validation pipeline for Layer 4."""

    def __init__(self):
        from src.shared.utils import load_yaml_config
        self._config = load_yaml_config("layer4_config.yaml")

    def test_message_formatter(self) -> dict:
        """Test MessageFormatter for all channels and scenarios."""
        from src.layer4_bots.message_formatter import MessageFormatter

        formatter = MessageFormatter(self._config)
        results = {"passed": 0, "failed": 0, "details": []}

        channels = {
            "whatsapp": self._config.get("message", {}).get("max_length_whatsapp", 280),
            "telegram_citizen": self._config.get("message", {}).get("max_length_telegram", 400),
            "dashboard": self._config.get("message", {}).get("max_length_dashboard", 600),
        }

        for decision in SAMPLE_DECISIONS:
            iid = decision["intersection_id"]
            for channel, max_chars in channels.items():
                issues = []
                t0 = time.time()
                msg = formatter.format_decision(decision, channel)
                elapsed_ms = int((time.time() - t0) * 1000)

                # Check length
                if len(msg) > max_chars:
                    issues.append(f"length={len(msg)} > max={max_chars}")

                # Check no forbidden terms (except dashboard which can have some)
                if channel != "dashboard":
                    for term in FORBIDDEN_TERMS:
                        if term.lower() in msg.lower():
                            issues.append(f"forbidden_term='{term}'")

                # Check intersection name present
                display_names = self._config.get("intersection_display_names", {})
                name = display_names.get(iid, "")
                if name and name not in msg and iid.replace("_", " ").title() not in msg:
                    issues.append("missing_intersection_name")

                # Check routes present for alta/critica
                if decision.get("severity") in ("alta", "critica") and decision.get("active_routes"):
                    has_route = any(r in msg for r in decision["active_routes"])
                    if not has_route and channel != "whatsapp":
                        issues.append("missing_routes_for_high_severity")

                if issues:
                    results["failed"] += 1
                else:
                    results["passed"] += 1

                results["details"].append({
                    "intersection": iid,
                    "channel": channel,
                    "length": len(msg),
                    "elapsed_ms": elapsed_ms,
                    "issues": issues,
                    "passed": len(issues) == 0,
                    "message_preview": msg[:80],
                })

        # Test batch formatting
        batch = formatter.format_batch(SAMPLE_DECISIONS, "whatsapp")
        # Should exclude "baja" (min severity = "media")
        severities = [b["severity"] for b in batch]
        if "baja" in severities:
            results["failed"] += 1
            results["details"].append({"test": "batch_filter", "passed": False, "issue": "baja included"})
        else:
            results["passed"] += 1
            results["details"].append({"test": "batch_filter", "passed": True})

        # Test system summary
        summary = formatter.format_system_summary(SAMPLE_FINAL_DECISION)
        if "Sogamoso" in summary and "31" in summary:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["details"].append({"test": "system_summary", "passed": False})

        return results

    def test_social_graph(self) -> dict:
        """Test SocialGraphModule with synthetic nodes."""
        from src.layer4_bots.social_graph import SocialGraphModule

        graph = SocialGraphModule(self._config)
        results = {"passed": 0, "failed": 0, "details": []}

        # Create 20 synthetic nodes
        corridors_pool = [
            ["carrera_11_norte", "carrera_11_sur"],
            ["av_castellana_entrada", "av_castellana_salida"],
            ["calle_14_centro_historico", "acceso_morca"],
            ["carrera_11_norte", "av_castellana_entrada"],
            ["carrera_11_sur", "calle_14_centro_historico"],
        ]
        roles = ["vecino", "conductor", "lider_barrio", "vecino", "conductor"]

        for i in range(20):
            graph.register_user({
                "user_id": f"test_user_{i:03d}",
                "corridors": corridors_pool[i % len(corridors_pool)],
                "peak_hours": [7, 8, 17, 18] if i % 2 == 0 else [6, 7, 12, 13],
                "query_count": i * 3,
                "is_seed": False,
                "role": roles[i % len(roles)],
            })

        # Test: graph has nodes
        n_nodes = graph._graph.number_of_nodes()
        if n_nodes >= 20:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["details"].append({"test": "node_count", "expected": ">=20", "got": n_nodes})

        # Test: compute centrality doesn't fail
        try:
            t0 = time.time()
            graph.compute_centrality()
            centrality_ms = int((time.time() - t0) * 1000)
            results["passed"] += 1
            results["details"].append({"test": "compute_centrality", "passed": True, "elapsed_ms": centrality_ms})
        except Exception as e:
            results["failed"] += 1
            results["details"].append({"test": "compute_centrality", "passed": False, "error": str(e)})

        # Test: scores in [0, 1]
        all_valid = True
        for uid, score in graph._propagator_ranking.items():
            if not (0.0 <= score <= 1.0):
                all_valid = False
                results["details"].append({"test": "score_bounds", "user": uid, "score": score})
                break
        if all_valid:
            results["passed"] += 1
        else:
            results["failed"] += 1

        # Test: get_propagators_for_corridor returns <= k
        propagators = graph.get_propagators_for_corridor(["carrera_11_norte", "carrera_11_sur"])
        k = self._config.get("social_graph", {}).get("first_wave_k", 5)
        # May expand k for coverage, but should be reasonable
        if len(propagators) <= k * 3:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["details"].append({"test": "propagator_count", "count": len(propagators), "max": k * 3})

        # Test: seed nodes have >= average score
        seed_scores = [graph._propagator_ranking.get(uid, 0) for uid in graph._graph.nodes
                       if graph._graph.nodes[uid].get("is_seed", False)]
        all_scores = list(graph._propagator_ranking.values())
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        seed_avg = sum(seed_scores) / len(seed_scores) if seed_scores else 0

        if seed_avg >= avg_score * 0.5:  # seeds should be at least somewhat central
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["details"].append({"test": "seed_centrality", "seed_avg": seed_avg, "graph_avg": avg_score})

        # Test: SIR simulation
        try:
            coverage = graph.simulate_sir_coverage(propagators[:3])
            if 0.0 <= coverage <= 1.0:
                results["passed"] += 1
                results["details"].append({"test": "sir_coverage", "coverage": coverage})
            else:
                results["failed"] += 1
        except Exception as e:
            results["failed"] += 1
            results["details"].append({"test": "sir_simulation", "error": str(e)})

        # Test: get_graph_summary
        summary = graph.get_graph_summary()
        if summary.get("total_users", 0) > 0:
            results["passed"] += 1
        else:
            results["failed"] += 1

        return results

    def test_alert_engine(self) -> dict:
        """Test AlertEngine filtering, dedup, and rate limiting."""
        from src.layer4_bots.alert_engine import AlertEngine

        engine = AlertEngine(self._config)
        results = {"passed": 0, "failed": 0, "details": []}

        # Test: severity filter
        baja_dec = {"intersection_id": "acceso_morca", "severity": "baja", "congestion_level": 0.2}
        alta_dec = {"intersection_id": "carrera_11_norte", "severity": "alta", "congestion_level": 0.8}

        if not engine._should_alert(baja_dec):
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["details"].append({"test": "filter_baja", "passed": False})

        if engine._should_alert(alta_dec):
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["details"].append({"test": "filter_alta", "passed": False})

        # Test: dedup
        engine._update_dedup("carrera_11_norte")
        if not engine._should_alert(alta_dec):
            results["passed"] += 1  # Should be deduped
        else:
            results["failed"] += 1
            results["details"].append({"test": "dedup", "passed": False})

        # Test: filter_and_sort
        filtered = engine._filter_and_sort_decisions(SAMPLE_DECISIONS)
        # "baja" should be excluded, "carrera_11_norte" should be deduped
        for d in filtered:
            if d.get("severity") == "baja":
                results["failed"] += 1
                results["details"].append({"test": "filter_sort_baja", "passed": False})
                break
        else:
            results["passed"] += 1

        # Test: rate limit
        engine2 = AlertEngine(self._config)
        for i in range(12):
            engine2._alert_timestamps.append(time.time())
        rate_dec = {"intersection_id": "acceso_morca_test", "severity": "critica", "congestion_level": 0.95}
        if not engine2._should_alert(rate_dec):
            results["passed"] += 1  # Rate limited
        else:
            results["failed"] += 1
            results["details"].append({"test": "rate_limit", "passed": False})

        return results

    def run_full_validation(self) -> dict:
        """Run all test suites."""
        logger.info("pipeline_layer4_starting")

        suites = {
            "message_formatter": self.test_message_formatter,
            "social_graph": self.test_social_graph,
            "alert_engine": self.test_alert_engine,
        }

        report = {"suites": {}, "total_passed": 0, "total_failed": 0}

        for name, test_fn in suites.items():
            logger.info("pipeline_suite_start", suite=name)
            t0 = time.time()
            try:
                result = test_fn()
                elapsed = int((time.time() - t0) * 1000)
                result["elapsed_ms"] = elapsed
                report["suites"][name] = result
                report["total_passed"] += result["passed"]
                report["total_failed"] += result["failed"]

                status = "✅ PASSED" if result["failed"] == 0 else "⚠️ PARTIAL"
                logger.info(
                    "pipeline_suite_done", suite=name, status=status,
                    passed=result["passed"], failed=result["failed"], elapsed_ms=elapsed,
                )
            except Exception as e:
                report["suites"][name] = {"passed": 0, "failed": 1, "error": str(e)}
                report["total_failed"] += 1
                logger.error("pipeline_suite_error", suite=name, error=str(e))

        report["all_passed"] = report["total_failed"] == 0
        return report


def main():
    """Run the validation pipeline."""
    print("=" * 60)
    print("TransitMind Sogamoso — Layer 4 Validation Pipeline")
    print("=" * 60)
    print()

    pipeline = Layer4Pipeline()
    report = pipeline.run_full_validation()

    print()
    print("=" * 60)
    total = report["total_passed"] + report["total_failed"]
    print(f"RESULTS: {report['total_passed']}/{total} tests passed")
    print("=" * 60)

    for suite_name, suite_result in report["suites"].items():
        p = suite_result.get("passed", 0)
        f = suite_result.get("failed", 0)
        ms = suite_result.get("elapsed_ms", 0)
        status = "✅" if f == 0 else "❌"
        print(f"  {status} {suite_name}: {p} passed, {f} failed ({ms}ms)")

        if f > 0:
            for detail in suite_result.get("details", []):
                if not detail.get("passed", True):
                    issues = detail.get("issues", []) or [detail.get("issue", detail.get("error", ""))]
                    print(f"     ⚠ {detail.get('test', detail.get('intersection', ''))}: {issues}")

    print()

    # Save report
    from src.shared.utils import get_project_root
    report_path = get_project_root() / "data" / "layer4_outputs" / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"Report saved to: {report_path}")

    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
