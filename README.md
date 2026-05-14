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

| Capa | Descripción | Estado |
|------|-------------|--------|
| **Capa 1: TimeGAN** | Generación de datos sintéticos de tráfico | ✅ Implementada |
| **Capa 2: LLM + RAG** | Análisis causal con modelos de lenguaje | 🔲 Placeholder |
| **Capa 3: Multi-Agentes** | 7 agentes orquestados con LangGraph | 🔲 Placeholder |
| **Capa 4: Bots & Dashboard** | Interfaz Telegram/WhatsApp + Streamlit | 🔲 Placeholder |

## 🏗️ Metodología

**TSTR (Train on Synthetic, Test on Real)**: Los modelos se entrenan con datos 100% sintéticos generados por TimeGAN y se validan con conteos reales de campo.

## 🚀 Inicio Rápido

```bash
# 1. Clonar y configurar
git clone https://github.com/EdwinMartinezGomez/TransitMind-Sogamoso.git
cd TransitMind-Sogamoso
cp .env.example .env

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar pipeline completo de Capa 1
make full-pipeline

# 4. Ver resultados en MLflow
make mlflow-ui
# → http://localhost:5000

# 5. Iniciar API
make api
# → http://localhost:8000/docs
```

## 📊 Intersecciones Piloto

- Carrera 11 Norte / Sur
- Avenida Castellana Entrada / Salida
- Calle 14 Centro Histórico
- Acceso Morca

## 📁 Estructura del Proyecto

```
src/
├── layer1_timegan/     # TimeGAN: data_loader, model, trainer, generator, evaluator, api
├── layer2_llm/         # (Placeholder) LLM + RAG
├── layer3_agents/      # (Placeholder) Multi-Agentes LangGraph
├── layer4_bots/        # (Placeholder) Telegram, WhatsApp, Dashboard
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

*TransitMind Sogamoso — Capa 1 TimeGAN | MLOps v1.0*  
*UPTC — Metodología TSTR*