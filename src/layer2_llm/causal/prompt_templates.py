"""
TransitMind Sogamoso — Prompt Templates
========================================
Structured prompts for causal traffic analysis via LLM.
"""

# ============================================
# System Prompt
# ============================================

SYSTEM_PROMPT = """Eres el Analista Causal de Tráfico para TransitMind Sogamoso.
Tu trabajo es analizar datos sintéticos de tráfico vehicular generados por un modelo TimeGAN
y contexto local de Sogamoso (Boyacá, Colombia) para producir un análisis causal estructurado.

REGLAS:
1. Responde SIEMPRE en JSON válido.
2. Responde SIEMPRE en español.
3. No incluyas comentarios, markdown ni texto fuera del JSON.
4. Basa tus conclusiones en los DATOS proporcionados y el CONTEXTO LOCAL.
5. Si no tienes suficiente información, indica confidence bajo (< 0.5).

ESTRUCTURA DE RESPUESTA (JSON):
{
  "causal_context": {
    "primary_cause": "string — causa principal de la congestión",
    "secondary_causes": ["lista de factores secundarios"],
    "causal_explanation": "string — explicación de cómo interactúan los factores",
    "severity": "baja|media|alta|critica",
    "confidence": 0.0-1.0
  },
  "traffic_forecast": {
    "congestion_level_adjusted": 0.0-1.0,
    "peak_window": "HH:MM - HH:MM",
    "expected_delay_minutes": 0-120,
    "affected_intersections": ["lista de IDs"]
  },
  "recommendations": {
    "traffic_light_adjustment": {
      "intersection_id": "string",
      "green_phase_extension_seconds": 0-60,
      "priority_direction": "norte_sur|este_oeste|rotacional",
      "rationale": "string"
    },
    "alternative_routes": ["lista de rutas alternativas"],
    "citizen_alert": "string — mensaje de alerta para ciudadanos (max 280 chars)"
  }
}"""

# ============================================
# User Prompt Template
# ============================================

USER_PROMPT_TEMPLATE = """Analiza el siguiente escenario de tráfico en Sogamoso:

## DATOS SINTÉTICOS (Layer 1 — TimeGAN)
- Intersección: {intersection_id} ({intersection_name})
- Hora: {hour}:00
- Hora pico: {is_peak_hour}
- Día de mercado: {is_market_day}
- Clima: {weather_description} (código: {weather_code})
- Flujo vehicular: {vehicle_flow:.1f} veh/15min
- Velocidad promedio: {avg_speed_kmh:.1f} km/h
- Congestión base: {congestion_level:.2f}
- Ratio vehículos pesados: {heavy_vehicle_ratio:.2f}
- Ratio motocicletas: {motorcycle_ratio:.2f}
- Impacto evento: {event_impact:.2f}

## CONTEXTO LOCAL (Knowledge Base RAG)
{rag_context}

## INSTRUCCIONES
Usando razonamiento causal paso a paso (chain-of-thought):
1. Identifica la CAUSA PRINCIPAL de congestión
2. Enumera factores SECUNDARIOS
3. Explica CÓMO interactúan los factores
4. Clasifica la SEVERIDAD
5. Genera RECOMENDACIONES concretas para Sogamoso

Responde SOLO con el JSON estructurado."""

# ============================================
# Few-Shot Examples
# ============================================

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": (
            "Intersección: carrera_11_norte. Hora: 7:00. Hora pico: True. "
            "Mercado: False. Clima: despejado. Congestión: 0.45. Evento: 0.0."
        ),
    },
    {
        "role": "assistant",
        "content": """{
  "causal_context": {
    "primary_cause": "hora pico",
    "secondary_causes": [],
    "causal_explanation": "La congestión moderada (0.45) se debe al flujo habitual de hora pico matutina. Sin factores agravantes como mercado o eventos, el patrón es predecible.",
    "severity": "baja",
    "confidence": 0.80
  },
  "traffic_forecast": {
    "congestion_level_adjusted": 0.50,
    "peak_window": "07:00 - 08:00",
    "expected_delay_minutes": 10,
    "affected_intersections": ["carrera_11_norte"]
  },
  "recommendations": {
    "traffic_light_adjustment": {
      "intersection_id": "carrera_11_norte",
      "green_phase_extension_seconds": 5,
      "priority_direction": "norte_sur",
      "rationale": "Extensión mínima para flujo matutino habitual."
    },
    "alternative_routes": ["Avenida Industrial", "Calle 14 (centro historico)"],
    "citizen_alert": "Tráfico normal en Carrera 11 Norte. Flujo habitual hora pico matutina."
  }
}""",
    },
    {
        "role": "user",
        "content": (
            "Intersección: calle_14_centro_historico. Hora: 6:00. Hora pico: False. "
            "Mercado: True. Clima: niebla (código 3). Congestión: 0.72. Evento: 0.1."
        ),
    },
    {
        "role": "assistant",
        "content": """{
  "causal_context": {
    "primary_cause": "mercado campesino",
    "secondary_causes": ["condiciones climaticas adversas", "carga rural temprana"],
    "causal_explanation": "La convergencia de mercado campesino en Plaza Central (miércoles/sábado, desde 4am) con niebla matutina de la cuenca del Chicamocha genera congestión alta (0.72). Los vehículos de carga rural reducen velocidad un 20-35% adicional por visibilidad limitada.",
    "severity": "alta",
    "confidence": 0.85
  },
  "traffic_forecast": {
    "congestion_level_adjusted": 0.82,
    "peak_window": "05:00 - 09:00",
    "expected_delay_minutes": 35,
    "affected_intersections": ["calle_14_centro_historico", "carrera_11_sur"]
  },
  "recommendations": {
    "traffic_light_adjustment": {
      "intersection_id": "calle_14_centro_historico",
      "green_phase_extension_seconds": 20,
      "priority_direction": "este_oeste",
      "rationale": "Priorizar flujo este-oeste para descongestionar acceso a Plaza Central durante mercado campesino."
    },
    "alternative_routes": ["Carrera 11", "Avenida Industrial"],
    "citizen_alert": "Alerta: congestión ALTA en Calle 14 por mercado campesino + niebla. Use Carrera 11 o Av Industrial. Reduzca velocidad."
  }
}""",
    },
]

# ============================================
# Repair Prompt (last-resort JSON fix)
# ============================================

REPAIR_PROMPT = """El siguiente texto debería ser un JSON válido pero tiene errores.
Corrige el JSON y devuelve SOLO el JSON corregido, sin explicaciones ni markdown.

Texto con errores:
{broken_json}

JSON corregido:"""
