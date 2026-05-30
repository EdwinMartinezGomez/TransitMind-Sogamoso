# src/layer4_bots/__init__.py
"""
Layer 4: Comunicación Ciudadana + Grafo Social
================================================
- message_formatter: Traduce JSON técnico → mensajes humanos (Ollama LLM)
- social_graph: Grafo G=(V,E,W) con betweenness + k-shell + SIR
- alert_engine: Orquestador de alertas con dedup y rate-limit
- telegram_bot: Bot de Telegram para ciudadanos y operadores
- whatsapp_handler: WhatsApp Business Cloud API handler
- dashboard: Dashboard Streamlit para Secretaría de Movilidad
- api: FastAPI en puerto 8003
"""
