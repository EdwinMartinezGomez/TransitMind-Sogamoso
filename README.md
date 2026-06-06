# TransitMind Sogamoso

[![CI](https://github.com/EdwinMartinezGomez/TransitMind-Sogamoso/actions/workflows/ci_train.yml/badge.svg)](https://github.com/EdwinMartinezGomez/TransitMind-Sogamoso/actions/workflows/ci_train.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-red.svg)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10%2B-blue.svg)](https://mlflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Sistema de inteligencia artificial para movilidad urbana orientado al análisis, simulación y comunicación de alertas de tráfico en Sogamoso, Colombia. El proyecto integra generación de datos sintéticos, análisis causal, coordinación multiagente y canales de notificación para usuarios ciudadanos y operadores.

## Resumen del proyecto

TransitMind Sogamoso está diseñado como una plataforma modular de cuatro capas:

| Capa | Propósito | Tecnologías principales |
|---|---|---|
| Capa 1: TimeGAN | Generación de datos sintéticos de tráfico para entrenamiento y experimentación | PyTorch, MLflow |
| Capa 2: LLM + RAG | Análisis causal y asistencia basada en conocimiento | Ollama, ChromaDB, LangChain |
| Capa 3: Multiagentes | Orquestación de agentes para evaluación y toma de decisiones | LangGraph, LangChain |
| Capa 4: Bots y dashboard | Difusión de alertas, interacción con usuarios y visualización | Telegram, FastAPI, NetworkX, Streamlit |

## Objetivo funcional

El sistema busca apoyar la gestión de la movilidad urbana mediante:

1. Generación de escenarios sintéticos consistentes con el comportamiento observado en campo.
2. Evaluación de decisiones de tráfico con modelos de lenguaje y reglas de negocio.
3. Coordinación de agentes especializados para monitoreo, predicción y recomendación.
4. Difusión priorizada de alertas a usuarios registrados a través de Telegram.

## Metodología

El proyecto sigue el enfoque TSTR (Train on Synthetic, Test on Real), donde los modelos se entrenan con datos sintéticos y se contrastan con conteos reales de referencia para validar consistencia y utilidad operativa.

## Componentes principales

| Componente | Ruta | Función |
|---|---|---|
| API Capa 1 | `src/layer1_timegan/` | Carga de datos, entrenamiento, generación y evaluación |
| API Capa 2 | `src/layer2_llm/` | Contexto, causalidad y recuperación de conocimiento |
| API Capa 3 | `src/layer3_agents/` | Agentes, estado del grafo y orquestación |
| API Capa 4 | `src/layer4_bots/` | Telegram, grafo social, alertas y dashboard |
| Utilidades compartidas | `src/shared/` | Logger, constantes, esquemas y helpers |

## Requisitos

- Python 3.10 o superior
- Dependencias instaladas desde `requirements.txt`
- Variables de entorno definidas en `.env`
- Token de Telegram para habilitar el bot

## Instalación y ejecución

```bash
git clone https://github.com/EdwinMartinezGomez/TransitMind-Sogamoso.git
cd TransitMind-Sogamoso
copy .env.example .env
pip install -r requirements.txt
```

En Windows PowerShell:

```powershell
.\run.ps1
```

Para ejecutar únicamente los servicios sin lanzar pipelines:

```powershell
.\run.ps1 -ServicesOnly
```

## Configuración mínima de Telegram

1. Crear el bot en [@BotFather](https://t.me/BotFather).
2. Guardar el token en `.env` como `TELEGRAM_BOT_TOKEN`.
3. Definir los identificadores de operadores en `TELEGRAM_OPERATOR_CHAT_IDS`.

## Servicios expuestos

| Servicio | Puerto | URL |
|---|---:|---|
| MLflow UI | 5000 | http://localhost:5000 |
| Capa 1 (TimeGAN) | 8000 | http://localhost:8000/docs |
| Capa 2 (LLM + RAG) | 8001 | http://localhost:8001/docs |
| Capa 3 (Agentes) | 8002 | http://localhost:8002/docs |
| Capa 4 (Bots API) | 8003 | http://localhost:8003/docs |
| Dashboard | 8501 | http://localhost:8501 |

## Grafo social y notificaciones

La Capa 4 construye un grafo social con NetworkX para priorizar la difusión de alertas. El módulo `src/layer4_bots/social_graph.py` registra usuarios, consultas y similitud entre corredores y horarios de actividad. Con esa información, `src/layer4_bots/alert_engine.py` calcula el orden de propagación y la API de Capa 4 coordina el envío de mensajes por Telegram.

## Intersecciones piloto

- Carrera 11 Norte / Sur
- Avenida Castellana Entrada / Salida
- Calle 14 Centro Histórico
- Acceso Morca

## Estructura del repositorio

```text
src/
├── layer1_timegan/     # Generación sintética, entrenamiento y evaluación
├── layer2_llm/         # Análisis causal, RAG y contexto
├── layer3_agents/      # Agentes, grafo de estado y orquestación
├── layer4_bots/        # Telegram, alertas, dashboard y grafo social
└── shared/             # Constantes, logging, utilidades y esquemas
docs/                   # Arquitectura, runbook y documentación técnica
configs/                # Configuración por capa
data/                    # Datos crudos, procesados y salidas de cada capa
experiments/            # Modelos, métricas y artefactos experimentales
tests/                  # Pruebas unitarias e integración
```

## Pruebas

```bash
make test
make test-cov
```

## Documentación adicional

- [Arquitectura del sistema](docs/arquitectura.md)
- [Variables de tráfico](docs/variables_trafico.md)
- [Protocolo TSTR](docs/tstr_protocol.md)
- [MLOps Runbook](docs/mlops_runbook.md)

## Estado del proyecto

El repositorio contiene implementaciones para las cuatro capas del sistema, pruebas automatizadas y documentación operativa. El foco principal es la simulación y priorización de alertas de movilidad urbana mediante datos sintéticos, análisis causal y mensajería ciudadana.