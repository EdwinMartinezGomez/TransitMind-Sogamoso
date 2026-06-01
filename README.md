# 🧠 TransitMind Sogamoso

> Sistema de IA para movilidad urbana con TimeGAN y Multi-Agentes  
> Universidad Pedagógica y Tecnológica de Colombia — UPTC

[![CI](https://github.com/EdwinMartinezGomez/TransitMind-Sogamoso/actions/workflows/ci_train.yml/badge.svg)](https://github.com/EdwinMartinezGomez/TransitMind-Sogamoso/actions/workflows/ci_train.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-red.svg)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10%2B-blue.svg)](https://mlflow.org/)

---

## 📋 Descripción

TransitMind Sogamoso es un sistema de inteligencia artificial para la gestión del tráfico urbano en Sogamoso, Colombia (120,000 habitantes). El proyecto utiliza una arquitectura de 4 capas:

| Capa | Descripción | Tecnología | Estado |
|------|-------------|-----------|--------|
| **Capa 1: TimeGAN** | Generación de datos sintéticos de tráfico | PyTorch, MLflow | ✅ Implementada |
| **Capa 2: LLM + RAG** | Análisis causal con modelos de lenguaje | Ollama, ChromaDB, LangChain | ✅ Implementada |
| **Capa 3: Multi-Agentes** | 7 agentes orquestados con LangGraph | LangGraph, LangChain | ✅ Implementada |
| **Capa 4: Bots & Dashboard** | Telegram/WhatsApp + Streamlit + Grafo Social | python-telegram-bot, NetworkX, Streamlit | ✅ Implementada |

## 🏗️ Metodología

**TSTR (Train on Synthetic, Test on Real)**: Los modelos se entrenan con datos 100% sintéticos generados por TimeGAN y se validan con conteos reales de campo.

## 🚀 Inicio Rápido

```bash
# 1. Clonar y configurar
git clone https://github.com/EdwinMartinezGomez/TransitMind-Sogamoso.git
cd TransitMind-Sogamoso
cp .env.example .env
# Editar .env con TELEGRAM_BOT_TOKEN (obtener de @BotFather)

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar todo (pipelines + servicios + bot)
.\run.ps1

# 4. Solo servicios (sin pipelines)
.\run.ps1 -ServicesOnly
```

## 🤖 Bot de Telegram

El bot permite a ciudadanos y operadores interactuar con el sistema:

### Comandos Ciudadanos
| Comando | Descripción |
|---------|-------------|
| `/start` | Registrarse y elegir corredor |
| `/estado` | Ver estado del tráfico en tiempo real |
| `/rutas` | Rutas alternativas activas |
| `/suscribir` | Suscribirse a un corredor |
| `/mizona` | Ver mis suscripciones |
| `/cancelar` | Dejar de recibir alertas |
| `/ayuda` | Ver todos los comandos |

### Comandos Operadores (Secretaría de Movilidad)
| Comando | Descripción |
|---------|-------------|
| `/ciclo` | Ejecutar ciclo de decisión |
| `/sistema` | Estado de todas las capas |
| `/reporte` | Resumen ejecutivo del último ciclo |

### Configuración
1. Crear bot en [@BotFather](https://t.me/BotFather)
2. Copiar el token en `.env` → `TELEGRAM_BOT_TOKEN`
3. Agregar IDs de operadores en `TELEGRAM_OPERATOR_CHAT_IDS`

## 🌐 Servicios

| Servicio | Puerto | URL |
|----------|--------|-----|
| MLflow UI | 5000 | http://localhost:5000 |
| Capa 1 (TimeGAN) | 8000 | http://localhost:8000/docs |
| Capa 2 (LLM+RAG) | 8001 | http://localhost:8001/docs |
| Capa 3 (Agentes) | 8002 | http://localhost:8002/docs |
| Capa 4 (Bots API) | 8003 | http://localhost:8003/docs |
| Dashboard | 8501 | http://localhost:8501 |
| Telegram Bot | — | Polling activo |

## 📊 Intersecciones Piloto

- Carrera 11 Norte / Sur
- Avenida Castellana Entrada / Salida
- Calle 14 Centro Histórico
- Acceso Morca

## 📁 Estructura del Proyecto

```
src/
├── layer1_timegan/     # TimeGAN: data_loader, model, trainer, generator, evaluator, api
├── layer2_llm/         # LLM + RAG: causal_analyst, rag_pipeline, context_builder, api
├── layer3_agents/      # Multi-Agentes: 7 agentes LangGraph + orchestrator, api
├── layer4_bots/        # Telegram Bot, WhatsApp, Dashboard Streamlit, Grafo Social
│   ├── telegram_bot.py       # Bot Telegram (ciudadanos + operadores)
│   ├── whatsapp_handler.py   # WhatsApp Business Cloud API
│   ├── alert_engine.py       # Motor de alertas con dedup + rate-limit
│   ├── social_graph.py       # Grafo G=(V,E,W) con betweenness + k-shell + SIR
│   ├── message_formatter.py  # JSON técnico → mensajes ciudadanos (Ollama)
│   ├── dashboard.py          # Dashboard Streamlit para SecMov
│   └── api.py                # FastAPI puerto 8003
└── shared/             # Logger, schemas, constants, utils
```

## 📚 Documentación

- [Arquitectura del Sistema](docs/arquitectura.md)
- [Variables de Tráfico](docs/variables_trafico.md)
- [Protocolo TSTR](docs/tstr_protocol.md)
- [MLOps Runbook](docs/mlops_runbook.md)

## 🧪 Tests

```bash
make test           # Unit + integration tests
make test-cov       # With coverage report
```

---

*TransitMind Sogamoso v1.0 — 4 Capas Completas*  
*UPTC — Inteligencia Computacional*