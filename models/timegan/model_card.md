# TransitMind Sogamoso — TimeGAN Model Card

## Resumen Ejecutivo
- **Estado**: ❌ NEEDS IMPROVEMENT
- **TSTR-Score**: 0.5979 (meta: ≥ 0.85)
- **Discriminative Accuracy**: 0.9987 (meta: ≤ 0.60)
- **JS Divergence Avg**: nan (meta: ≤ 0.15)
- **Correlation Diff**: nan (meta: ≤ 0.10)

## Métricas TSTR
| Métrica | Sintético | Real |
|---------|-----------|------|
| MAE | 0.0326 | 0.0232 |
| RMSE | 0.0514 | 0.0281 |
| R² | 0.928 | 0.9785 |

## Discriminative Score
- **Accuracy**: 0.9987
- **AUC-ROC**: 1.0

## Criterios de Aceptación
| Métrica | Valor | Umbral Mín | Umbral Ideal | Estado |
|---------|-------|------------|--------------|--------|
| TSTR-score | 0.5979 | ≥ 0.80 | ≥ 0.90 | ❌ |
| Disc. Accuracy | 0.9987 | ≤ 0.60 | ≤ 0.52 | ❌ |
| JS Divergence | nan | ≤ 0.15 | ≤ 0.08 | ❌ |
| Correlation Diff | nan | ≤ 0.10 | ≤ 0.05 | ❌ |

## Recomendaciones
Se recomienda ajustar hiperparámetros y re-entrenar el modelo.

---
*Generado automáticamente por TransitMind Sogamoso — TSTR Evaluator*
