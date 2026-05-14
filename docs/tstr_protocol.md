# Protocolo TSTR — Train on Synthetic, Test on Real

## Descripción

El protocolo TSTR valida la calidad de los datos sintéticos generados por TimeGAN comparando el rendimiento de modelos entrenados con datos sintéticos vs datos reales, ambos evaluados sobre datos reales de campo.

## Flujo

1. **Entrenar predictor LSTM en datos SINTÉTICOS**
2. **Evaluar predictor en datos REALES** → obtener MAE_synthetic
3. **Entrenar mismo predictor en datos REALES** (baseline)
4. **Evaluar baseline en datos REALES** → obtener MAE_real
5. **Calcular TSTR-score** = 1 - |MAE_synthetic - MAE_real| / MAE_real

## Criterios de Aceptación

| Métrica | Umbral Mínimo | Umbral Ideal |
|---------|--------------|--------------|
| TSTR-score | ≥ 0.80 | ≥ 0.90 |
| Discriminative accuracy | ≤ 0.60 | ≤ 0.52 |
| JS Divergence avg | ≤ 0.15 | ≤ 0.08 |
| Correlation diff | ≤ 0.10 | ≤ 0.05 |

## Referencia

Esteban et al. (2023) "Real-valued medical time series generation with recurrent conditional GANs and the TSTR evaluation protocol"
