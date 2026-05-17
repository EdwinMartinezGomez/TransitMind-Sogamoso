"""
TransitMind Sogamoso — Pydantic Schemas
========================================
Data validation schemas used across all layers.
Ensures consistent data contracts between components.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
from typing import Literal


# ============================================
# Traffic Data Schemas
# ============================================

class TrafficRecord(BaseModel):
    """Schema for a single traffic observation record."""

    timestamp: datetime = Field(..., description="Observation timestamp")
    hour: int = Field(..., ge=0, le=23, description="Hour of day")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    is_peak_hour: bool = Field(..., description="Peak hour flag")
    vehicle_flow: float = Field(..., ge=0, le=300, description="Vehicle flow (veh/15min)")
    heavy_vehicle_ratio: float = Field(..., ge=0.0, le=1.0, description="Heavy vehicle ratio")
    motorcycle_ratio: float = Field(..., ge=0.0, le=1.0, description="Motorcycle ratio")
    avg_speed_kmh: float = Field(..., ge=0, le=80, description="Average speed (km/h)")
    congestion_level: float = Field(..., ge=0.0, le=1.0, description="Congestion level (TARGET)")
    weather_code: int = Field(..., ge=0, le=3, description="Weather condition code")
    event_impact: float = Field(..., ge=0.0, le=1.0, description="Event impact level")
    is_market_day: bool = Field(..., description="Market day flag")
    intersection_id: str = Field(..., description="Intersection identifier")


class TrafficSequence(BaseModel):
    """Schema for a sequence of traffic observations (TimeGAN input)."""

    intersection_id: str
    records: List[TrafficRecord]
    seq_len: int = Field(default=24, ge=1)


# ============================================
# API Schemas
# ============================================

class GenerateRequest(BaseModel):
    """Request schema for the /generate endpoint."""

    intersection_id: str = Field(
        ...,
        description="Target intersection ID",
        examples=["carrera_11_norte"],
    )
    n_samples: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Number of synthetic samples to generate",
    )
    scenario: Optional[str] = Field(
        default=None,
        description="Predefined scenario name (e.g., 'market_day', 'morning_fog')",
    )
    scenario_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom scenario parameters",
    )

    @field_validator("intersection_id")
    @classmethod
    def validate_intersection(cls, v: str) -> str:
        valid_ids = [
            "carrera_11_norte",
            "carrera_11_sur",
            "av_castellana_entrada",
            "av_castellana_salida",
            "calle_14_centro_historico",
            "acceso_morca",
        ]
        if v not in valid_ids:
            raise ValueError(f"Invalid intersection_id. Must be one of: {valid_ids}")
        return v


class GenerateResponse(BaseModel):
    """Response schema for the /generate endpoint."""

    synthetic_data: List[Dict[str, Any]] = Field(
        ..., description="Generated synthetic traffic records"
    )
    metadata: Dict[str, Any] = Field(
        ..., description="Generation metadata (scenario, timestamp, count)"
    )


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    last_training_date: Optional[str] = Field(
        None, description="Last training date (ISO format)"
    )
    version: str = Field(default="1.0.0", description="API version")


# ============================================
# Agent Interface Schemas
# ============================================

class AgentTrafficData(BaseModel):
    """
    Schema for data exchange between Layer 1 and Layer 3 agents.
    Used by export_for_agents() in the generator module.
    """

    intersection_id: str
    timestamp: str
    vehicle_flow: float
    congestion_level: float
    avg_speed_kmh: float
    weather_code: int
    scenario_metadata: Optional[Dict[str, Any]] = None


class AgentDataBatch(BaseModel):
    """Batch of traffic data for agent consumption."""

    data: List[AgentTrafficData]
    scenario_name: str
    generated_at: str
    n_samples: int
    schema_version: str = "1.0"


# ============================================
# Evaluation Schemas
# ============================================

class EvaluationMetrics(BaseModel):
    """Schema for TSTR evaluation results."""

    tstr_score: float = Field(..., ge=0.0, le=1.0)
    discriminative_accuracy: float = Field(..., ge=0.0, le=1.0)
    discriminative_auc: float = Field(..., ge=0.0, le=1.0)
    js_divergence_avg: float = Field(..., ge=0.0)
    correlation_diff: float = Field(..., ge=0.0)
    mae_synthetic: float = Field(..., ge=0.0)
    mae_real: float = Field(..., ge=0.0)
    rmse_synthetic: float = Field(..., ge=0.0)
    rmse_real: float = Field(..., ge=0.0)
    r2_synthetic: float
    r2_real: float
    passed: bool = Field(..., description="Whether all acceptance thresholds are met")


# ============================================
# Layer 2: Causal Analysis Schemas
# ============================================

class CausalContext(BaseModel):
    """Contextual causal information about traffic congestion."""

    primary_cause: str = Field(..., description="Primary cause of congestion")
    secondary_causes: List[str] = Field(default_factory=list, description="Secondary factors")
    causal_explanation: str = Field(default="", description="How factors interact")
    severity: Literal["baja", "media", "alta", "critica"] = Field(
        ..., description="Severity level"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Analysis confidence")


class TrafficForecast(BaseModel):
    """Adjusted traffic forecast with causal context."""

    congestion_level_adjusted: float = Field(..., ge=0.0, le=1.0)
    peak_window: str = Field(..., description="Expected peak window (HH:MM - HH:MM)")
    expected_delay_minutes: int = Field(..., ge=0, le=120)
    affected_intersections: List[str] = Field(default_factory=list)


class TrafficLightAdjustment(BaseModel):
    """Recommended traffic light timing adjustment."""

    intersection_id: str
    green_phase_extension_seconds: int = Field(..., ge=0, le=60)
    priority_direction: Literal["norte_sur", "este_oeste", "rotacional"] = Field(
        ..., description="Priority direction for green phase"
    )
    rationale: str = Field(default="", description="Reason for this adjustment")


class Recommendations(BaseModel):
    """Actionable recommendations for traffic management."""

    traffic_light_adjustment: TrafficLightAdjustment
    alternative_routes: List[str] = Field(default_factory=list)
    citizen_alert: str = Field(
        ..., max_length=280, description="Public alert message (max WhatsApp/Telegram)"
    )


class CausalAnalysisResult(BaseModel):
    """Complete causal analysis result from Layer 2."""

    intersection_id: str
    analysis_timestamp: datetime
    causal_context: CausalContext
    traffic_forecast: TrafficForecast
    recommendations: Recommendations
    rag_sources_used: List[str] = Field(default_factory=list)
    llm_reasoning_trace: str = Field(default="")
    processing_time_ms: int = Field(default=0, ge=0)
    model_used: str = Field(default="fallback")
    is_fallback: bool = Field(default=False)


# ============================================
# Layer 2: API Request / Response Schemas
# ============================================

class AnalyzeRequest(BaseModel):
    """Request for POST /analyze — receives full Layer 1 response."""

    intersection_id: str = Field(default="carrera_11_norte")
    scenario: Optional[str] = Field(default=None)
    generated_at: Optional[str] = Field(default=None)
    n_samples: int = Field(default=100, ge=1)
    synthetic_data: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class AnalyzeScenarioRequest(BaseModel):
    """Request for POST /analyze-scenario — orchestrates Layer 1 + Layer 2."""

    intersection_id: str = Field(
        ...,
        description="Target intersection ID",
        examples=["carrera_11_norte"],
    )
    scenario: str = Field(
        ...,
        description="Scenario name (e.g., 'market_day', 'morning_fog')",
        examples=["market_day"],
    )
    n_samples: int = Field(default=50, ge=1, le=10000)


class AnalyzeAllRequest(BaseModel):
    """Request for POST /analyze-all — all intersections for a scenario."""

    scenario: str = Field(..., description="Scenario name")
    n_samples: int = Field(default=50, ge=1, le=10000)


class Layer2HealthResponse(BaseModel):
    """Response for GET /health on Layer 2."""

    status: Literal["ok", "degraded", "error"]
    llm_available: bool = Field(default=False)
    chroma_docs_count: int = Field(default=0)
    last_analysis_timestamp: Optional[str] = Field(default=None)
    layer1_api_reachable: bool = Field(default=False)


class KnowledgeBaseStatus(BaseModel):
    """Response for GET /knowledge-base/status."""

    documents_indexed: int = Field(default=0)
    collections: List[str] = Field(default_factory=list)
    last_ingestion: Optional[str] = Field(default=None)
    embedding_model: str = Field(default="nomic-embed-text")

