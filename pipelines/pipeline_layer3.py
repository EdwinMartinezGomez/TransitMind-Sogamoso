"""
TransitMind Sogamoso — Layer 3 Validation Pipeline
=====================================================
Validates that Layer 3 meets acceptance criteria by running
all 6 Sogamoso scenarios and checking key metrics.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.shared.logger import get_logger, setup_logger

setup_logger("pipeline_layer3", level="INFO")
logger = get_logger("pipeline_layer3")

# All 6 Sogamoso scenarios
SCENARIOS = [
    "normal_weekday",
    "market_day",
    "morning_fog",
    "coliseo_event",
    "rain_market",
    "duitama_blockage",
]

# Acceptance criteria
ACCEPTANCE_CRITERIA = {
    "min_tmc_reduction": 0.25,
    "max_cycle_duration_ms": 120_000,
    "expected_agents_healthy": 7,
    "expected_decisions_count": 6,
}


class Layer3Pipeline:
    """Validation pipeline for Layer 3 multi-agent system."""

    def run_full_validation(self) -> dict:
        """
        Execute all 6 scenarios and validate acceptance criteria.

        Returns:
            Validation report dict.
        """
        from src.layer3_agents.orchestrator import run_cycle

        logger.info("pipeline_starting", scenarios=len(SCENARIOS))

        results = {}
        passed_count = 0
        total_tmc = 0.0
        total_duration = 0

        for scenario in SCENARIOS:
            logger.info("pipeline_scenario_start", scenario=scenario)
            t0 = time.time()

            try:
                result = run_cycle(scenario=scenario)
                elapsed_ms = int((time.time() - t0) * 1000)

                # Validate criteria
                issues = []
                tmc_reduction = result.get("global_tmc_reduction_percent", 0.0) / 100.0
                monitor_report = result.get("monitor_report", {})
                decisions = result.get("decisions", [])
                agents_healthy = monitor_report.get("agents_healthy", 0)
                cycle_ms = monitor_report.get("cycle_duration_ms", elapsed_ms)

                if tmc_reduction < ACCEPTANCE_CRITERIA["min_tmc_reduction"]:
                    issues.append(
                        f"tmc_reduction={tmc_reduction:.2%} < {ACCEPTANCE_CRITERIA['min_tmc_reduction']:.0%}"
                    )

                if cycle_ms > ACCEPTANCE_CRITERIA["max_cycle_duration_ms"]:
                    issues.append(
                        f"cycle_duration={cycle_ms}ms > {ACCEPTANCE_CRITERIA['max_cycle_duration_ms']}ms"
                    )

                if agents_healthy != ACCEPTANCE_CRITERIA["expected_agents_healthy"]:
                    issues.append(
                        f"agents_healthy={agents_healthy} != {ACCEPTANCE_CRITERIA['expected_agents_healthy']}"
                    )

                if len(decisions) != ACCEPTANCE_CRITERIA["expected_decisions_count"]:
                    issues.append(
                        f"decisions={len(decisions)} != {ACCEPTANCE_CRITERIA['expected_decisions_count']}"
                    )

                scenario_passed = len(issues) == 0
                if scenario_passed:
                    passed_count += 1

                total_tmc += tmc_reduction
                total_duration += cycle_ms

                results[scenario] = {
                    "passed": scenario_passed,
                    "tmc_reduction": round(tmc_reduction, 4),
                    "duration_ms": cycle_ms,
                    "decisions_count": len(decisions),
                    "agents_healthy": agents_healthy,
                    "issues": issues,
                }

                status_str = "✅ PASSED" if scenario_passed else "❌ FAILED"
                logger.info(
                    "pipeline_scenario_done",
                    scenario=scenario,
                    status=status_str,
                    tmc_reduction=f"{tmc_reduction:.2%}",
                    duration_ms=cycle_ms,
                    issues=issues,
                )

            except Exception as e:
                elapsed_ms = int((time.time() - t0) * 1000)
                results[scenario] = {
                    "passed": False,
                    "tmc_reduction": 0.0,
                    "duration_ms": elapsed_ms,
                    "decisions_count": 0,
                    "agents_healthy": 0,
                    "issues": [f"Exception: {str(e)}"],
                }
                logger.error(
                    "pipeline_scenario_exception",
                    scenario=scenario,
                    error=str(e),
                )

        n_scenarios = len(SCENARIOS)
        report = {
            "scenarios_tested": n_scenarios,
            "passed": passed_count,
            "failed": n_scenarios - passed_count,
            "avg_tmc_reduction": round(total_tmc / max(n_scenarios, 1), 4),
            "avg_cycle_duration_ms": int(total_duration / max(n_scenarios, 1)),
            "details": results,
        }

        # Print summary
        logger.info(
            "pipeline_complete",
            passed=f"{passed_count}/{n_scenarios}",
            avg_tmc=f"{report['avg_tmc_reduction']:.2%}",
            avg_duration_ms=report["avg_cycle_duration_ms"],
        )

        return report

    def run_stress_test(self, n_cycles: int = 5) -> dict:
        """
        Execute n_cycles of the same scenario consecutively.

        Verifies:
        - Graph doesn't accumulate state between cycles
        - Results are consistent (don't degrade)
        - No memory leaks evident
        """
        from src.layer3_agents.orchestrator import run_cycle

        logger.info("stress_test_starting", n_cycles=n_cycles)

        cycle_results = []
        scenario = "normal_weekday"

        for i in range(n_cycles):
            t0 = time.time()
            try:
                result = run_cycle(scenario=scenario)
                elapsed_ms = int((time.time() - t0) * 1000)

                tmc = result.get("global_tmc_reduction_percent", 0.0)
                decisions = len(result.get("decisions", []))

                cycle_results.append({
                    "cycle_index": i + 1,
                    "tmc_reduction_percent": tmc,
                    "decisions_count": decisions,
                    "duration_ms": elapsed_ms,
                    "has_error": "error" in result,
                })

                logger.info(
                    "stress_cycle_done",
                    cycle=i + 1,
                    tmc=f"{tmc:.1f}%",
                    duration_ms=elapsed_ms,
                )

            except Exception as e:
                elapsed_ms = int((time.time() - t0) * 1000)
                cycle_results.append({
                    "cycle_index": i + 1,
                    "tmc_reduction_percent": 0.0,
                    "decisions_count": 0,
                    "duration_ms": elapsed_ms,
                    "has_error": True,
                    "error": str(e),
                })

        # Analyze consistency
        tmc_values = [c["tmc_reduction_percent"] for c in cycle_results if not c.get("has_error")]
        duration_values = [c["duration_ms"] for c in cycle_results if not c.get("has_error")]

        report = {
            "n_cycles": n_cycles,
            "successful_cycles": len(tmc_values),
            "failed_cycles": n_cycles - len(tmc_values),
            "tmc_min": min(tmc_values) if tmc_values else 0,
            "tmc_max": max(tmc_values) if tmc_values else 0,
            "tmc_avg": sum(tmc_values) / len(tmc_values) if tmc_values else 0,
            "duration_avg_ms": int(sum(duration_values) / len(duration_values)) if duration_values else 0,
            "cycles": cycle_results,
        }

        logger.info(
            "stress_test_complete",
            successful=report["successful_cycles"],
            tmc_avg=f"{report['tmc_avg']:.1f}%",
            duration_avg=report["duration_avg_ms"],
        )

        return report


def main():
    """Run the full validation pipeline."""
    pipeline = Layer3Pipeline()

    print("=" * 60)
    print("TransitMind Sogamoso — Layer 3 Validation Pipeline")
    print("=" * 60)
    print()

    report = pipeline.run_full_validation()

    print()
    print("=" * 60)
    print(f"RESULTS: {report['passed']}/{report['scenarios_tested']} scenarios passed")
    print(f"Average TMC Reduction: {report['avg_tmc_reduction']:.2%}")
    print(f"Average Cycle Duration: {report['avg_cycle_duration_ms']}ms")
    print("=" * 60)

    for scenario, detail in report["details"].items():
        status = "✅" if detail["passed"] else "❌"
        print(f"  {status} {scenario}: TMC={detail['tmc_reduction']:.2%}, {detail['duration_ms']}ms")
        if detail["issues"]:
            for issue in detail["issues"]:
                print(f"     ⚠ {issue}")

    print()

    # Save report
    from src.shared.utils import get_project_root

    report_path = get_project_root() / "data" / "layer3_outputs" / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Report saved to: {report_path}")

    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
