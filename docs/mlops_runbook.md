# MLOps Runbook — TransitMind Sogamoso

## Comandos Rápidos

```bash
# Generar datos semilla
make generate-data

# Entrenar TimeGAN
make train

# Evaluar con TSTR
make evaluate

# Pipeline completo
make full-pipeline

# Ver MLflow UI
make mlflow-ui      # → http://localhost:5000

# API
make api            # → http://localhost:8000

# Tests
make test

# Docker
make docker-up      # Levanta MLflow + PostgreSQL + API
make docker-down
```

## Estructura de Experimentos MLflow

- **Experimento**: `transitmind-timegan-layer1`
  - Run: `phase_a_autoencoder` — Reconstruction loss
  - Run: `phase_b_supervisor` — Supervised loss
  - Run: `phase_c_joint_training` — D-loss, G-loss, Recon-loss
  - Run: `tstr_evaluation` — TSTR-score, discriminative accuracy

## Troubleshooting

| Problema | Solución |
|----------|----------|
| Mode collapse (G-loss → 0) | Reducir lr_generator, aumentar gamma |
| Reconstruction loss no baja | Aumentar epochs_autoencoder, verificar datos |
| TSTR-score < 0.80 | Más epochs en Fase C, ajustar hidden_dim |
| OOM en GPU | Reducir batch_size, usar CPU |
