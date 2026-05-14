"""
TransitMind Sogamoso — Pydantic Schemas
========================================
Data validation schemas used across all layers.
Ensures consistent data contracts between components.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


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
