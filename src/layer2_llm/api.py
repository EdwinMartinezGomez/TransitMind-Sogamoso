"""
TransitMind Sogamoso — Layer 2 API
===================================
FastAPI REST API for LLM-powered causal traffic analysis.
Runs on port 8001, consumes Layer 1 data and returns structured analysis.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.shared.logger import get_logger
from src.shared.schemas import (
	AnalyzeRequest,
	AnalyzeScenarioRequest,
	AnalyzeAllRequest,
	CausalAnalysisResult,
	KnowledgeBaseStatus,
	Layer2HealthResponse,
)
from src.shared.utils import load_yaml_config
from src.layer2_llm.causal_analyst import CausalAnalyst
from src.layer2_llm.rag_pipeline import RagPipeline

logger = get_logger("layer2.api")

# Load config
config = load_yaml_config("llm_config.yaml")
layer1_cfg = config.get("layer1_api", {})
LAYER1_BASE = layer1_cfg.get("base_url", "http://localhost:8000")
LAYER1_TIMEOUT = int(layer1_cfg.get("timeout_seconds", 15))

# Initialize components
analyst = CausalAnalyst(config)
rag_pipeline = RagPipeline(config)

# Create FastAPI app
app = FastAPI(
	title="TransitMind Sogamoso — Layer 2 API",
	description="LLM + RAG causal traffic analysis for Sogamoso, Boyacá",
	version="1.0.0",
)

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Track last analysis timestamp
_last_analysis_ts: str | None = None


# ============================================
# Endpoints
# ============================================

@app.post("/analyze", response_model=CausalAnalysisResult)
async def analyze(request: AnalyzeRequest):
	"""
	Analyze Layer 1 data and return causal analysis.
	Receives synthetic traffic data directly.
	"""
	global _last_analysis_ts

	try:
		payload = request.model_dump()
		result = analyst.analyze(payload)
		_last_analysis_ts = datetime.utcnow().isoformat()
		return result
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error("analyze_failed", error=str(e))
		raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze-scenario", response_model=CausalAnalysisResult)
async def analyze_scenario(request: AnalyzeScenarioRequest):
	"""
	Orchestrate Layer 1 + Layer 2: generate synthetic data, then analyze.
	"""
	global _last_analysis_ts

	# Step 1: Call Layer 1 to generate synthetic data
	try:
		async with httpx.AsyncClient(timeout=LAYER1_TIMEOUT) as client:
			layer1_response = await client.post(
				f"{LAYER1_BASE}/generate",
				json={
					"intersection_id": request.intersection_id,
					"scenario": request.scenario,
					"n_samples": request.n_samples,
				},
			)
			layer1_response.raise_for_status()
			layer1_data = layer1_response.json()
	except httpx.ConnectError:
		raise HTTPException(
			status_code=503,
			detail=f"Layer 1 API not reachable at {LAYER1_BASE}",
		)
	except httpx.HTTPStatusError as e:
		raise HTTPException(
			status_code=502,
			detail=f"Layer 1 returned error: {e.response.status_code}",
		)
	except Exception as e:
		raise HTTPException(
			status_code=502,
			detail=f"Layer 1 call failed: {str(e)}",
		)

	# Step 2: Analyze with Layer 2
	try:
		result = analyst.analyze(layer1_data)
		_last_analysis_ts = datetime.utcnow().isoformat()
		return result
	except Exception as e:
		logger.error("scenario_analysis_failed", error=str(e))
		raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze-all")
async def analyze_all(request: AnalyzeAllRequest):
	"""
	Analyze all intersections for a given scenario.
	"""
	global _last_analysis_ts

	intersections = [
		item["id"]
		for item in config.get("intersections", [])
		if isinstance(item, dict) and "id" in item
	]

	results = []
	errors = []

	for intersection_id in intersections:
		try:
			async with httpx.AsyncClient(timeout=LAYER1_TIMEOUT) as client:
				layer1_response = await client.post(
					f"{LAYER1_BASE}/generate",
					json={
						"intersection_id": intersection_id,
						"scenario": request.scenario,
						"n_samples": request.n_samples,
					},
				)
				layer1_response.raise_for_status()
				layer1_data = layer1_response.json()

			result = analyst.analyze(layer1_data)
			results.append(result.model_dump())
		except Exception as e:
			errors.append({"intersection_id": intersection_id, "error": str(e)})
			logger.warning("analyze_all_failed", intersection=intersection_id, error=str(e))

	_last_analysis_ts = datetime.utcnow().isoformat()

	return {
		"scenario": request.scenario,
		"total_intersections": len(intersections),
		"successful": len(results),
		"failed": len(errors),
		"results": results,
		"errors": errors,
	}


@app.get("/health", response_model=Layer2HealthResponse)
async def health():
	"""Check Layer 2 health: LLM, ChromaDB, Layer 1 connectivity."""
	# Check LLM
	llm_available = False
	try:
		import ollama as ollama_client
		ollama_client.list()
		llm_available = True
	except Exception:
		pass

	# Check ChromaDB
	chroma_count = 0
	try:
		chroma_count = rag_pipeline._retriever._vector_store.get_collection_count()
	except Exception:
		pass

	# Check Layer 1
	layer1_reachable = False
	try:
		async with httpx.AsyncClient(timeout=5) as client:
			resp = await client.get(f"{LAYER1_BASE}/health")
			layer1_reachable = resp.status_code == 200
	except Exception:
		pass

	status = "ok"
	if not llm_available:
		status = "degraded"
	if chroma_count == 0 and not llm_available:
		status = "error"

	return Layer2HealthResponse(
		status=status,
		llm_available=llm_available,
		chroma_docs_count=chroma_count,
		last_analysis_timestamp=_last_analysis_ts,
		layer1_api_reachable=layer1_reachable,
	)


@app.get("/knowledge-base/status", response_model=KnowledgeBaseStatus)
async def kb_status():
	"""Get knowledge base status."""
	try:
		count = rag_pipeline._retriever._vector_store.get_collection_count()
		collections = rag_pipeline._retriever._vector_store.get_collections_list()
		rag_cfg = config.get("rag", {})

		return KnowledgeBaseStatus(
			documents_indexed=count,
			collections=collections,
			embedding_model=rag_cfg.get("embedding_model", "nomic-embed-text"),
		)
	except Exception as e:
		logger.error("kb_status_failed", error=str(e))
		return KnowledgeBaseStatus()


@app.post("/knowledge-base/refresh")
async def kb_refresh(background_tasks: BackgroundTasks):
	"""Re-ingest knowledge base documents in background."""

	def _do_refresh():
		try:
			result = rag_pipeline._ingestion.ingest_all(force_reingest=True)
			logger.info("kb_refresh_complete", result=result)
		except Exception as e:
			logger.error("kb_refresh_failed", error=str(e))

	background_tasks.add_task(_do_refresh)
	return {"status": "refresh_started", "message": "Knowledge base re-indexing started in background."}


# ============================================
# Startup event
# ============================================

@app.on_event("startup")
async def startup():
	"""Initialize knowledge base on startup."""
	logger.info("layer2_api_starting")
	try:
		result = rag_pipeline.ingest_if_needed()
		logger.info("startup_ingestion", result=result)
	except Exception as e:
		logger.warning("startup_ingestion_failed", error=str(e))


if __name__ == "__main__":
	import uvicorn

	api_cfg = config.get("layer2_api", {})
	uvicorn.run(
		"src.layer2_llm.api:app",
		host=api_cfg.get("host", "0.0.0.0"),
		port=api_cfg.get("port", 8001),
		reload=api_cfg.get("reload", False),
	)
