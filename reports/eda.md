# EDA — Customer Churn

## Dataset

- Filas: **7,043**
- Columnas: **21**
- Duplicados exactos: **0**
- Target: `Churn`
- `Churn=Yes`: **26.37%** (1,857)
- `Churn=No`: **73.63%** (5,186)
- Faltantes: `TotalCharges` tiene **26** valores faltantes; no hay faltantes en las demás columnas.
- `customerID` es un identificador y se excluye del modelado.

## Preparación

Las variables numéricas (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`) se imputan por mediana y escalan. Las categóricas se imputan por moda y se transforman con OneHotEncoder. Todo forma parte del mismo `ColumnTransformer`/`Pipeline`, de modo que entrenamiento e inferencia comparten transformaciones.

## Variables de negocio observadas

La tasa de churn por contrato es descriptivamente diferente entre categorías en este dataset sintético. Se utiliza como señal exploratoria, no como regla de decisión.

| Contract | Proporción Churn=Yes |
|---|---:|
| Month-to-month | 38.73% |
| One year | 14.97% |
| Two year | 8.97% |

## Evaluación

La consigna exige más que accuracy: se reportan precision, recall, F1, ROC-AUC y matriz de confusión. Dado el costo operativo potencial de no detectar a un cliente que abandona, el análisis presta especial atención a `recall` y a los falsos negativos.
