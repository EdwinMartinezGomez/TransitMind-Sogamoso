"""
TransitMind Sogamoso — Layer 2 Integration Pipeline
=====================================================
Tests the Layer 2 LLM + RAG pipeline end-to-end.
"""

import sys
import time
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.shared.logger import get_logger
from src.shared.utils import load_yaml_config

logger = get_logger("pipeline.layer2")


def run_ingestion_test(config):
    """Test knowledge base document ingestion."""
    print("\n" + "=" * 60)
    print("  STEP 1: Knowledge Base Ingestion")
    print("=" * 60)

    from src.layer2_llm.rag.document_ingestion import DocumentIngestion

    ingestion = DocumentIngestion(config)

    # Force re-ingest for testing
    result = ingestion.ingest_all(force_reingest=True)
    print(f"  Status: {result.get('status')}")
    print(f"  Documents loaded: {result.get('documents_loaded', 0)}")
    print(f"  Chunks created: {result.get('chunks_created', 0)}")
    print(f"  Chunks indexed: {result.get('chunks_indexed', 0)}")
    print(f"  Ollama embeddings: {result.get('ollama_used', False)}")
    print(f"  Time: {result.get('elapsed_seconds', 0)}s")

    if result.get("status") == "error":
        print(f"  ERROR: {result.get('message')}")
        return False

    return True


def run_retrieval_test(config):
    """Test RAG retrieval with sample queries."""
    print("\n" + "=" * 60)
    print("  STEP 2: RAG Retrieval Test")
    print("=" * 60)

    from src.layer2_llm.rag_pipeline import RagPipeline

    rag = RagPipeline(config)

    test_payloads = [
        {
            "name": "market_day",
            "intersection_id": "calle_14_centro_historico",
            "synthetic_data": [{
                "congestion_level": 0.7,
                "is_market_day": True,
                "is_peak_hour": False,
                "weather_code": 0,
                "event_impact": 0.0,
                "hour": 6,
                "vehicle_flow": 150,
                "avg_speed_kmh": 20,
                "heavy_vehicle_ratio": 0.15,
                "motorcycle_ratio": 0.3,
            }],
        },
        {
            "name": "fog_morning",
            "intersection_id": "acceso_morca",
            "synthetic_data": [{
                "congestion_level": 0.5,
                "is_market_day": False,
                "is_peak_hour": True,
                "weather_code": 3,
                "event_impact": 0.0,
                "hour": 7,
                "vehicle_flow": 100,
                "avg_speed_kmh": 25,
                "heavy_vehicle_ratio": 0.1,
                "motorcycle_ratio": 0.2,
            }],
        },
    ]

    all_ok = True
    for payload in test_payloads:
        name = payload.pop("name")
        result = rag.build_context(payload, intersection_id=payload["intersection_id"])
        query = result.get("query", "")
        context = result.get("context", "")
        sources = result.get("sources", [])

        print(f"\n  Scenario: {name}")
        print(f"  Query preview: {query[:80]}...")
        print(f"  Context length: {len(context)} chars")
        print(f"  Sources: {sources}")

        if not context:
            print("  WARNING: No context retrieved")
            all_ok = False

    return all_ok


def run_analysis_test(config):
    """Test full causal analysis (LLM or fallback)."""
    print("\n" + "=" * 60)
    print("  STEP 3: Causal Analysis Test")
    print("=" * 60)

    from src.layer2_llm.causal_analyst import CausalAnalyst

    analyst = CausalAnalyst(config)

    test_scenarios = [
        {
            "name": "Normal traffic",
            "intersection_id": "carrera_11_norte",
            "synthetic_data": [{
                "congestion_level": 0.3,
                "is_market_day": False,
                "is_peak_hour": False,
                "weather_code": 0,
                "event_impact": 0.0,
                "hour": 10,
                "vehicle_flow": 80,
                "avg_speed_kmh": 35,
                "heavy_vehicle_ratio": 0.05,
                "motorcycle_ratio": 0.2,
                "intersection_id": "carrera_11_norte",
            }],
        },
        {
            "name": "Market day + fog",
            "intersection_id": "calle_14_centro_historico",
            "synthetic_data": [{
                "congestion_level": 0.75,
                "is_market_day": True,
                "is_peak_hour": False,
                "weather_code": 3,
                "event_impact": 0.1,
                "hour": 6,
                "vehicle_flow": 160,
                "avg_speed_kmh": 15,
                "heavy_vehicle_ratio": 0.2,
                "motorcycle_ratio": 0.25,
                "intersection_id": "calle_14_centro_historico",
            }],
        },
        {
            "name": "Major event",
            "intersection_id": "calle_14_centro_historico",
            "synthetic_data": [{
                "congestion_level": 0.85,
                "is_market_day": False,
                "is_peak_hour": True,
                "weather_code": 0,
                "event_impact": 0.9,
                "hour": 19,
                "vehicle_flow": 200,
                "avg_speed_kmh": 10,
                "heavy_vehicle_ratio": 0.1,
                "motorcycle_ratio": 0.15,
                "intersection_id": "calle_14_centro_historico",
            }],
        },
    ]

    all_ok = True
    for scenario in test_scenarios:
        name = scenario.pop("name")
        start = time.time()

        try:
            result = analyst.analyze(scenario)
            elapsed = time.time() - start

            print(f"\n  Scenario: {name}")
            print(f"  Intersection: {result.intersection_id}")
            print(f"  Primary cause: {result.causal_context.primary_cause}")
            print(f"  Severity: {result.causal_context.severity}")
            print(f"  Confidence: {result.causal_context.confidence:.2f}")
            print(f"  Congestion adjusted: {result.traffic_forecast.congestion_level_adjusted:.2f}")
            print(f"  Delay (min): {result.traffic_forecast.expected_delay_minutes}")
            print(f"  Model: {result.model_used}")
            print(f"  Is fallback: {result.is_fallback}")
            print(f"  Processing time: {result.processing_time_ms}ms")
            print(f"  RAG sources: {result.rag_sources_used}")
            print(f"  Alert: {result.recommendations.citizen_alert}")

        except Exception as e:
            print(f"\n  Scenario: {name}")
            print(f"  ERROR: {e}")
            all_ok = False

    return all_ok


def run_llm_check():
    """Check if Ollama LLM is available."""
    print("\n" + "=" * 60)
    print("  STEP 0: LLM Availability Check")
    print("=" * 60)

    try:
        import ollama as ollama_client
        models = ollama_client.list()

        model_names = []
        if hasattr(models, "models"):
            model_names = [m.model for m in models.models]
        elif isinstance(models, dict):
            model_names = [m.get("name", "") for m in models.get("models", [])]

        print(f"  Ollama is running")
        print(f"  Available models: {model_names}")
        return True

    except ImportError:
        print("  Ollama Python package not installed")
        return False
    except Exception as e:
        print(f"  Ollama not available: {e}")
        return False


def main():
    print("=" * 60)
    print("  TransitMind Sogamoso — Layer 2 Integration Test")
    print("=" * 60)

    config = load_yaml_config("llm_config.yaml")

    # Step 0: Check LLM
    llm_ok = run_llm_check()

    # Step 1: Ingestion
    ingestion_ok = run_ingestion_test(config)
    if not ingestion_ok:
        print("\n  FAILED: Ingestion test")
        sys.exit(1)

    # Step 2: Retrieval
    retrieval_ok = run_retrieval_test(config)

    # Step 3: Analysis
    analysis_ok = run_analysis_test(config)

    # Summary
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  LLM available:     {'YES' if llm_ok else 'NO (using fallback)'}")
    print(f"  Ingestion:         {'PASS' if ingestion_ok else 'FAIL'}")
    print(f"  Retrieval:         {'PASS' if retrieval_ok else 'WARN'}")
    print(f"  Analysis:          {'PASS' if analysis_ok else 'FAIL'}")
    print("=" * 60)

    if not analysis_ok:
        sys.exit(1)

    print("\n  All Layer 2 tests passed!")


if __name__ == "__main__":
    main()
