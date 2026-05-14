"""
TransitMind Sogamoso — FastAPI Endpoint (Layer 1 ↔ Layer 3 Interface)
======================================================================
Exposes the TimeGAN generator as a REST API for the multi-agent system.
"""

from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.shared.schemas import GenerateRequest, GenerateResponse, HealthResponse
from src.shared.logger import setup_logger

logger = setup_logger("api")

app = FastAPI(
    title="TransitMind Sogamoso — Layer 1 API",
    description="Synthetic traffic data generation via TimeGAN",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global generator instance (lazy loaded)
_generator = None
_last_training_date: Optional[str] = None


def _get_generator():
    """Lazy-load the TrafficDataGenerator."""
    global _generator, _last_training_date
    if _generator is None:
        try:
            from src.layer1_timegan.generator import TrafficDataGenerator
            _generator = TrafficDataGenerator()
            _last_training_date = datetime.now().isoformat()
        except Exception as e:
            logger.error("generator_init_failed", error=str(e))
            raise HTTPException(status_code=503, detail=f"Model not available: {e}")
    return _generator


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    try:
        gen = _get_generator()
        model_loaded = gen.model_loaded
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        last_training_date=_last_training_date,
        version="1.0.0",
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate synthetic traffic data."""
    gen = _get_generator()

    if not gen.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first.")

    from src.layer1_timegan.generator import SOGAMOSO_SCENARIOS

    scenario_params = request.scenario_params
    if request.scenario and request.scenario in SOGAMOSO_SCENARIOS:
        s = SOGAMOSO_SCENARIOS[request.scenario]
        scenario_params = {k: v for k, v in s.items() if k != "description"}

    try:
        df = gen.generate(
            n_samples=request.n_samples,
            intersection_id=request.intersection_id,
            scenario=scenario_params,
        )
        records = df.to_dict(orient="records")
        # Convert timestamps to strings
        for r in records:
            if "timestamp" in r:
                r["timestamp"] = str(r["timestamp"])

        return GenerateResponse(
            synthetic_data=records,
            metadata={
                "intersection_id": request.intersection_id,
                "n_samples": request.n_samples,
                "scenario": request.scenario,
                "generated_at": datetime.now().isoformat(),
                "n_records": len(records),
            },
        )
    except Exception as e:
        logger.error("generation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
