"""
TransitMind Sogamoso — Output Parser
======================================
Robust JSON extraction from LLM responses with cascading strategies.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from src.shared.logger import get_logger

logger = get_logger("layer2.output_parser")


class OutputParser:
    """Extracts and validates JSON from LLM text responses."""

    def parse(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to extract valid JSON from LLM response using cascading strategies.

        Strategy order:
          1. Direct json.loads()
          2. Extract ```json ... ``` block
          3. Extract first { ... } block
          4. Clean up common issues (trailing commas, comments)

        Args:
            text: Raw LLM response text.

        Returns:
            Parsed dict or None if all strategies fail.
        """
        if not text or not text.strip():
            return None

        text = text.strip()

        # Strategy 1: Direct parse
        result = self._try_direct(text)
        if result is not None:
            logger.debug("parse_strategy", strategy="direct")
            return result

        # Strategy 2: Extract markdown JSON block
        result = self._try_markdown_block(text)
        if result is not None:
            logger.debug("parse_strategy", strategy="markdown_block")
            return result

        # Strategy 3: Extract first { ... } block
        result = self._try_brace_extraction(text)
        if result is not None:
            logger.debug("parse_strategy", strategy="brace_extraction")
            return result

        # Strategy 4: Clean and retry
        result = self._try_clean_and_parse(text)
        if result is not None:
            logger.debug("parse_strategy", strategy="cleaned")
            return result

        logger.warning("all_parse_strategies_failed", text_preview=text[:200])
        return None

    def validate_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and fill missing fields with safe defaults.

        Args:
            data: Parsed JSON dict from LLM.

        Returns:
            Validated dict with all required fields.
        """
        # Ensure top-level keys exist
        if "causal_context" not in data:
            data["causal_context"] = {}
        if "traffic_forecast" not in data:
            data["traffic_forecast"] = {}
        if "recommendations" not in data:
            data["recommendations"] = {}

        # Fill causal_context defaults
        ctx = data["causal_context"]
        ctx.setdefault("primary_cause", "flujo vehicular habitual")
        ctx.setdefault("secondary_causes", [])
        ctx.setdefault("causal_explanation", "Análisis generado por LLM.")
        ctx.setdefault("severity", "media")
        ctx.setdefault("confidence", 0.5)

        # Validate severity
        valid_severities = {"baja", "media", "alta", "critica"}
        if ctx["severity"] not in valid_severities:
            ctx["severity"] = "media"

        # Clamp confidence
        ctx["confidence"] = max(0.0, min(1.0, float(ctx["confidence"])))

        # Fill traffic_forecast defaults
        fc = data["traffic_forecast"]
        fc.setdefault("congestion_level_adjusted", 0.5)
        fc.setdefault("peak_window", "07:00 - 08:00")
        fc.setdefault("expected_delay_minutes", 10)
        fc.setdefault("affected_intersections", [])

        # Clamp forecast values
        fc["congestion_level_adjusted"] = max(
            0.0, min(1.0, float(fc["congestion_level_adjusted"]))
        )
        fc["expected_delay_minutes"] = max(
            0, min(120, int(fc["expected_delay_minutes"]))
        )

        # Fill recommendations defaults
        rec = data["recommendations"]
        if "traffic_light_adjustment" not in rec:
            rec["traffic_light_adjustment"] = {}

        adj = rec["traffic_light_adjustment"]
        adj.setdefault("intersection_id", "")
        adj.setdefault("green_phase_extension_seconds", 10)
        adj.setdefault("priority_direction", "rotacional")
        adj.setdefault("rationale", "Ajuste sugerido por análisis LLM.")

        # Validate priority_direction
        valid_dirs = {"norte_sur", "este_oeste", "rotacional"}
        if adj["priority_direction"] not in valid_dirs:
            adj["priority_direction"] = "rotacional"

        # Clamp extension
        adj["green_phase_extension_seconds"] = max(
            0, min(60, int(adj["green_phase_extension_seconds"]))
        )

        rec.setdefault("alternative_routes", [])

        # Citizen alert — ensure it exists and is max 280 chars
        rec.setdefault("citizen_alert", "Revise condiciones de tráfico antes de salir.")
        if len(rec["citizen_alert"]) > 280:
            rec["citizen_alert"] = rec["citizen_alert"][:277].rstrip() + "..."

        return data

    # ============================================
    # Parse strategies
    # ============================================

    def _try_direct(self, text: str) -> Optional[Dict[str, Any]]:
        """Strategy 1: Direct JSON parse."""
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def _try_markdown_block(self, text: str) -> Optional[Dict[str, Any]]:
        """Strategy 2: Extract ```json ... ``` blocks."""
        patterns = [
            r"```json\s*\n(.*?)\n\s*```",
            r"```\s*\n(.*?)\n\s*```",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group(1).strip())
                    if isinstance(result, dict):
                        return result
                except (json.JSONDecodeError, ValueError):
                    pass
        return None

    def _try_brace_extraction(self, text: str) -> Optional[Dict[str, Any]]:
        """Strategy 3: Find the outermost { ... } block."""
        first_brace = text.find("{")
        if first_brace == -1:
            return None

        # Find matching closing brace
        depth = 0
        last_brace = -1
        for i in range(first_brace, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    last_brace = i
                    break

        if last_brace == -1:
            return None

        candidate = text[first_brace : last_brace + 1]
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def _try_clean_and_parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Strategy 4: Clean common issues and retry."""
        # Extract { ... } first
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace == -1 or last_brace == -1:
            return None

        candidate = text[first_brace : last_brace + 1]

        # Remove single-line comments
        candidate = re.sub(r"//.*?$", "", candidate, flags=re.MULTILINE)

        # Remove trailing commas before } or ]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

        # Remove control characters
        candidate = re.sub(r"[\x00-\x1f\x7f]", " ", candidate)

        # Fix unquoted keys (simple heuristic)
        candidate = re.sub(
            r"(?<=\{|,)\s*(\w+)\s*:", r' "\1":', candidate
        )

        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        return None
