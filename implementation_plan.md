# TransitMind Sogamoso — Capa 2 (LLM + RAG) Implementation Plan

## Descripción General

Implementar la Capa 2 completa del proyecto TransitMind Sogamoso: un pipeline LLM + RAG que recibe datos sintéticos de Capa 1 (TimeGAN) y produce análisis causal estructurado para consumo por Capa 3 (Agentes).

**Estado actual**: La carpeta `src/layer2_llm/` tiene 4 archivos placeholder (`__init__.py`, `causal_analyst.py`, `context_builder.py`, `rag_pipeline.py`) — todos vacíos con TODOs.

**Estado objetivo**: ~30 archivos funcionales organizados en subpaquetes `rag/`, `causal/`, `connectors/`, `knowledge_base/` con API REST en puerto 8001.

---

## User Review Required

> [!IMPORTANT]
> **Proveedor de LLM**: El config usa Ollama con `llama3:8b` y `nomic-embed-text` como default. La implementación asume que **Ollama está instalado y ambos modelos están descargados** en la máquina local. Si prefieres usar **Groq** (gratuito, basado en nube, más rápido), se puede cambiar via `configs/llm_config.yaml`. ¿Qué proveedor deseas usar como principal?

> [!WARNING]  
> **Archivos placeholder existentes**: Los archivos `causal_analyst.py`, `context_builder.py`, `rag_pipeline.py` serán **eliminados** (ya que son stubs vacíos) y reemplazados por la nueva estructura modular. `__init__.py` será reescrito.

> [!IMPORTANT]
> **Dependencias pesadas**: Se agregarán ~12 dependencias nuevas al `requirements.txt` (langchain, chromadb, httpx, pymupdf, etc.) sumando ~500MB+ al environment. La implementación usa `sentence-transformers` para embeddings cuando Ollama no está disponible como fallback.

---

## Open Questions

> [!IMPORTANT]
> 1. **¿Ollama ya está instalado?** Si no, el sistema funcionará en modo fallback (reglas deterministas sin LLM). ¿Quieres que incluya instrucciones de instalación de Ollama en la documentación?
> 2. **¿Los PDFs mencionados** (`plan_movilidad_sogamoso_2023.pdf`, `calendario_ferias_boyaca_2024.pdf`) **existen realmente** o debo generar solo los archivos `.txt` de la knowledge base? El prompt menciona ambos pero solo pide crear los `.txt`.
> 3. **¿Quieres que las dependencias de Capa 2 se instalen inmediatamente** (`pip install`) o solo se agreguen al `requirements.txt` para instalación manual posterior?

---

## Proposed Changes

### Fase 0 — Configuración y Estructura Base

#### [NEW] [llm_config.yaml](file:///c:/Users/edwin/OneDrive/Escritorio/Trabajos%20U/Trabajos%20Noveno%20Semestre/IA/TransitMind-Sogamoso/configs/llm_config.yaml)
- Configuración completa de LLM (provider, model, temperature), RAG (embeddings, ChromaDB, chunk size), umbrales de análisis causal, endpoints de Capa 1/2, e intersecciones piloto.

#### [MODIFY] [requirements.txt](file:///c:/Users/edwin/OneDrive/Escritorio/Trabajos%20U/Trabajos%20Noveno%20Semestre/IA/TransitMind-Sogamoso/requirements.txt)
- Agregar dependencias de Capa 2: `langchain`, `langchain-community`, `chromadb`, `httpx`, `pymupdf`, `python-dateutil`, `groq`.
- Mantener las dependencias existentes de Capa 1 intactas.

#### [MODIFY] [pyproject.toml](file:///c:/Users/edwin/OneDrive/Escritorio/Trabajos%20U/Trabajos%20Noveno%20Semestre/IA/TransitMind-Sogamoso/pyproject.toml)
- Agregar grupo de dependencias opcionales `llm` con las mismas dependencias.

#### [MODIFY] [schemas.py](file:///c:/Users/edwin/OneDrive/Escritorio/Trabajos%20U/Trabajos%20Noveno%20Semestre/IA/TransitMind-Sogamoso/src/shared/schemas.py)
- Agregar modelos Pydantic: `CausalContext`, `TrafficForecast`, `TrafficLightAdjustment`, `Recommendations`, `CausalAnalysisResult`, `AnalyzeRequest`, `AnalyzeScenarioRequest`, `Layer2HealthResponse`, `KnowledgeBaseStatus`.

#### [DELETE] [causal_analyst.py](file:///c:/Users/edwin/OneDrive/Escritorio/Trabajos%20U/Trabajos%20Noveno%20Semestre/IA/TransitMind-Sogamoso/src/layer2_llm/causal_analyst.py) (stub)
#### [DELETE] [context_builder.py](file:///c:/Users/edwin/OneDrive/Escritorio/Trabajos%20U/Trabajos%20Noveno%20Semestre/IA/TransitMind-Sogamoso/src/layer2_llm/context_builder.py) (stub)
#### [DELETE] [rag_pipeline.py](file:///c:/Users/edwin/OneDrive/Escritorio/Trabajos%20U/Trabajos%20Noveno%20Semestre/IA/TransitMind-Sogamoso/src/layer2_llm/rag_pipeline.py) (stub)
#### [MODIFY] [\_\_init\_\_.py](file:///c:/Users/edwin/OneDrive/Escritorio/Trabajos%20U/Trabajos%20Noveno%20Semestre/IA/TransitMind-Sogamoso/src/layer2_llm/__init__.py)
- Actualizar para exportar los módulos principales del nuevo layout.

---

### Fase 0b — Knowledge Base Documents

#### [NEW] `src/layer2_llm/knowledge_base/documents/mercados_campesinos_schedule.txt`
Schedule detallado de mercados campesinos: días (mi/sá), horarios (5am-2pm), ubicaciones (Plaza Central, Plaza de Mercado Cubierta), impacto por intersección, flujo de carga rural desde 4am.

#### [NEW] `src/layer2_llm/knowledge_base/documents/patrones_niebla_chicamocha.txt`
Comportamiento niebla cuenca Chicamocha: frecuencia (40% oct-feb), horarios (5am-9am), intersecciones afectadas, reducción velocidad (-20% a -35%), protocolo alerta.

#### [NEW] `src/layer2_llm/knowledge_base/documents/rutas_alternativas_sogamoso.txt`
Rutas alternativas por intersección piloto: ruta primaria, alternativas A/B, tiempo adicional, tipo de vía. Incluye Carrera 9, Av Industrial, variante Morca.

#### [NEW] `src/layer2_llm/knowledge_base/documents/eventos_coliseo_olimpico.txt`
Patrones congestión Coliseo Olímpico: radio impacto (500m), horarios críticos, impacto por tipo evento, ajuste semáforos Calle 14.

#### [NEW] `src/layer2_llm/knowledge_base/documents/flujos_intermunicipales.txt`
Patrones buses intermunicipales: terminales, horarios pico arribo, intersecciones impactadas (Av Castellana), frecuencia temporada alta.

---

### Fase 1 — RAG Pipeline (Ingestion + Vector Store)

#### [NEW] `src/layer2_llm/rag/__init__.py`
#### [NEW] `src/layer2_llm/rag/document_ingestion.py`
- Clase `DocumentIngestion`: `load_documents()`, `chunk_documents()`, `embed_and_store()`, `ingest_all()`.
- Soporta `.txt` y `.pdf` (PyMuPDF). Chunks de 500 tokens, overlap 50.
- Embeddings via Ollama `nomic-embed-text` (fallback: `sentence-transformers`).
- Metadata incluye `filename`, `doc_type`, `intersection_relevance`.

#### [NEW] `src/layer2_llm/rag/vector_store.py`
- Clase `VectorStoreManager`: gestión de ChromaDB (crear/obtener colección, agregar/consultar documentos, status).
- Persistencia en `src/layer2_llm/knowledge_base/chroma_db/`.

#### [NEW] `src/layer2_llm/rag/retriever.py`
- Clase `ContextRetriever`: `retrieve()`, `format_context()`, `retrieve_with_context()`.
- Threshold de similitud 0.65, top_k=4, max 2000 tokens contexto.

#### [NEW] `src/layer2_llm/rag/query_builder.py`
- Clase `QueryBuilder`: `build_query()`, `build_intersection_filter()`.
- Transforma JSON numérico de Capa 1 → query semántica en español.
- Mapeo: `is_market_day=True` → "mercado campesino plaza central", etc.

---

### Fase 2 — Prompt Templates

#### [NEW] `src/layer2_llm/causal/__init__.py`
#### [NEW] `src/layer2_llm/causal/prompt_templates.py`
- `SYSTEM_PROMPT`: Rol del Analista Causal TransitMind.
- `USER_PROMPT_TEMPLATE`: Template principal con chain-of-thought, datos de tráfico, contexto RAG.
- `FEW_SHOT_EXAMPLES`: 2 ejemplos (caso normal, caso mercado+niebla).
- `SEVERITY_CLASSIFICATION_PROMPT`, `ROUTE_RECOMMENDATION_PROMPT`, `CITIZEN_ALERT_PROMPT`, `MULTI_INTERSECTION_PROMPT`.
- `REPAIR_PROMPT`: Para reparación de JSON malformado.

#### [NEW] `src/layer2_llm/causal/context_builder.py`
- Clase `ContextBuilder`: Combina datos sintéticos + contexto RAG + datos de conectores externos → prompt completo poblado.

---

### Fase 3 — Analista Causal (Núcleo)

#### [NEW] `src/layer2_llm/causal/causal_analyst.py`
- Clase `CausalAnalyst`: Pipeline principal `analyze()` de 7 pasos:
  1. Preprocesar datos Capa 1 (estadísticas, weather_code → texto)
  2. Recuperar contexto RAG
  3. Construir prompt con templates + few-shot
  4. Invocar LLM (Ollama/Groq, timeout 30s)
  5. Parsear y validar JSON
  6. Enriquecer con metadatos
  7. Persistir y loguear
- `analyze_batch()`: Análisis paralelo con asyncio.
- `_fallback_analysis()`: Reglas deterministas cuando el LLM falla.

---

### Fase 4 — Output Parser

#### [NEW] `src/layer2_llm/causal/output_parser.py`
- Clase `OutputParser`: Estrategia en cascada (4 intentos) para extraer JSON válido.
  - Intento 1: `json.loads()` directo
  - Intento 2: Extraer bloque `{...}`
  - Intento 3: Limpiar markdown/comments/trailing commas
  - Intento 4: Regex para reconstruir JSON mínimo
- `validate_schema()`: Validación contra Pydantic con defaults seguros.
- `repair_with_llm()`: Último recurso, prompt de reparación al LLM.

---

### Fase 5 — Conectores Externos

#### [NEW] `src/layer2_llm/connectors/__init__.py`
#### [NEW] `src/layer2_llm/connectors/layer1_client.py`
- Clase `Layer1Client` con httpx async: `generate_scenario()`, `get_all_intersections_data()`.
- Retry 2x con backoff exponencial, timeout 15s.

#### [NEW] `src/layer2_llm/connectors/ideam_client.py`
- Clase `IdeamWeatherClient`: Lookup table estática de patrones climáticos de Sogamoso.
- `get_weather_description()`: Convierte weather_code → texto descriptivo.

#### [NEW] `src/layer2_llm/connectors/calendar_client.py`
- Clase `EventCalendarClient`: `get_active_events()`, `is_special_date()`.
- Detecta ferias, Semana Santa, festivos nacionales, temporada navideña.

---

### Fase 6 — API FastAPI

#### [NEW] `src/layer2_llm/api.py`
- FastAPI app en puerto 8001 con 6 endpoints:
  1. `POST /analyze` — Recibe datos Capa 1, retorna análisis causal
  2. `POST /analyze-scenario` — Orquesta Capa 1 + Capa 2
  3. `POST /analyze-all` — Todas las intersecciones para un escenario
  4. `GET /health` — Estado de LLM, ChromaDB, Capa 1
  5. `GET /knowledge-base/status` — Status de base vectorial
  6. `POST /knowledge-base/refresh` — Re-indexación en background
- CORS, rate limiting simple, error handling JSON.

#### [MODIFY] [Makefile](file:///c:/Users/edwin/OneDrive/Escritorio/Trabajos%20U/Trabajos%20Noveno%20Semestre/IA/TransitMind-Sogamoso/Makefile)
- Agregar targets: `layer2-ingest`, `layer2-api`, `layer2-test`.

---

### Fase 7 — Pipeline Integrado y Tests

#### [NEW] `pipelines/pipeline_layer2.py`
- Clase `Layer2Pipeline`: `run_integration_test()`, `run_consistency_check()`.
- Valida 6 escenarios con criterios de aceptación (response time, JSON parse rate, confidence, consistency).

#### [NEW] `tests/integration/test_layer2.py`
- 6 tests de integración: market_day, fog, event, fallback, JSON schema, full flow.

#### [NEW] `data/layer2_outputs/causal_analyses/.gitkeep`
#### [NEW] `data/layer2_outputs/rag_queries_log/.gitkeep`

---

## Resumen de Archivos

| Acción | Cantidad | Ubicación |
|--------|----------|-----------|
| **Nuevos** | ~25 archivos | `src/layer2_llm/`, `configs/`, `data/`, `pipelines/`, `tests/` |
| **Modificados** | 5 archivos | `requirements.txt`, `pyproject.toml`, `schemas.py`, `Makefile`, `__init__.py` |
| **Eliminados** | 3 stubs | `causal_analyst.py`, `context_builder.py`, `rag_pipeline.py` (los placeholder vacíos) |

---

## Verification Plan

### Automated Tests
```bash
# 1. Verificar que la API inicia sin errores
python -c "from src.layer2_llm.api import app; print('API module loads OK')"

# 2. Verificar schemas Pydantic
python -c "from src.shared.schemas import CausalAnalysisResult; print('Schemas OK')"

# 3. Verificar ingestion de documentos
python -c "from src.layer2_llm.rag.document_ingestion import DocumentIngestion; print('Ingestion OK')"

# 4. Ejecutar tests unitarios
python -m pytest tests/ -v --tb=short

# 5. Verificar que knowledge base docs existen y tienen contenido
python -c "from pathlib import Path; docs=list(Path('src/layer2_llm/knowledge_base/documents').glob('*.txt')); print(f'{len(docs)} documents found')"
```

### Manual Verification
- Iniciar API Capa 2 con `make layer2-api` y verificar `GET /health`
- Verificar que `POST /knowledge-base/refresh` indexa documentos correctamente
- Si Ollama está disponible: probar `POST /analyze-scenario` con escenario `market_day`
- Si Ollama NO está disponible: verificar que fallback determinista funciona correctamente
