# Selección del modelo candidato

## Modelo

**LogisticRegression(C=1.0)** dentro de un Pipeline de scikit-learn.

## Motivo de selección

Con el split estratificado de 80/20 y `random_state=42`, la corrida candidata obtuvo:

- Precision: **0.6640**
- Recall: **0.4516**
- F1: **0.5376**
- ROC-AUC: **0.8120**
- Falsos negativos: **204**
- Falsos positivos: **85**

El baseline de mayoría no detecta positivos y sirve como referencia; el baseline estratificado también queda muy por debajo en métricas discriminativas. Los Random Forest evaluados ofrecen distinta relación precision/recall, pero en este split presentan menor recall y F1 que `logistic_c1`. Por eso esta corrida se documenta como **modelo candidato** para la siguiente etapa.

## Limitación

La selección corresponde únicamente al split experimental definido para la Entrega 1. No implica una garantía de performance en datos futuros. La revisión de drift sobre datos de producción queda reservada para la etapa final del proyecto.
