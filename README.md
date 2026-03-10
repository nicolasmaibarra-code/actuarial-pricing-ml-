# Pricing Actuarial con GLM y Scoring Comercial con ML

Modelo de pricing actuarial de punta a punta sobre el dataset **freMTPL2** (seguro automotor de responsabilidad civil francés). El proyecto sigue la metodología frecuencia-severidad estándar en la industria aseguradora, incorpora una capa de scoring comercial con XGBoost, y cuantifica el impacto de eliminar la variable BonusMalus mediante un ablation study.

---

## Metodología

La prima pura se descompone en dos modelos independientes:

```
Prima Pura = E[ClaimNb | X] × E[ClaimAmount | ClaimNb > 0, X]
           = GLM Poisson (frecuencia) × GLM Gamma (severidad)
```

Sobre la prima base del GLM se aplica una capa de scoring comercial (XGBoost):

```
Prima Scored = Prima Pura × Score de Riesgo Relativo
```

El score se normaliza a media = 1.0 sobre el conjunto de entrenamiento, preservando la suficiencia de prima a nivel de cartera.

---

## Dataset

**Fuente:** [freMTPL2 — OpenML](https://www.openml.org/d/41214)

| Dataset | Registros | Variables |
|---|---|---|
| freMTPL2freq (pólizas) | 678.013 | 12 |
| freMTPL2sev (siniestros) | 26.639 | 2 |

26.444 siniestros matcheados tras el join (195 siniestros huérfanos excluidos, 0,73% del total).

---

## Especificación de los Modelos

### GLM Poisson — Frecuencia

```
log(E[ClaimNb]) = log(Exposure) + β₀ + β₁·VehAge + β₂·DrivAge + β₃·BonusMalus
               + β₄·VehPower + β₅·Density + Σγₖ·1[VehGas=k] + Σδⱼ·1[Region=j]
```

- Exposición incorporada como offset (coeficiente restringido a 1.0)
- Variables categóricas via dummies con categoría de referencia (`smf.glm`)

### GLM Gamma — Severidad

```
log(E[ClaimAmount | ClaimNb > 0]) = β₀ + β₁·VehAge + β₂·DrivAge + β₃·BonusMalus
```

- Especificación parsimoniosa dado el tamaño reducido del dataset (~26k siniestros vs ~678k pólizas)
- Entrenado exclusivamente sobre pólizas con ClaimNb > 0

### XGBoost — Scoring Comercial

- Entrenado sobre el conjunto completo de features (9 variables incluyendo categóricas)
- Target: predicciones crudas del GLM Poisson en frecuencia
- Score = predicción XGBoost / media(predicciones XGBoost en train)

---

## Resultados

### Ablation Study — Impacto de BonusMalus

| Modelo | Poisson Deviance | Degradación |
|---|---|---|
| GLM — Con BonusMalus | 0,3218 | — |
| GLM — Sin BonusMalus | 0,3282 | +1,99% |
| XGBoost — Con BonusMalus | 0,3076 | — |
| XGBoost — Sin BonusMalus | 0,3136 | +1,93% |

BonusMalus es un score actuarial precalculado que puede no estar disponible en contextos de despliegue insurtech. El ablation study cuantifica su impacto y motiva la búsqueda de proxies alternativos (telemetría, datos de comportamiento, score crediticio).

### Observaciones clave

- El GLM fue elegido sobre las alternativas de ML por parsimonia: la diferencia de performance no justifica el costo en interpretabilidad en un contexto de pricing regulado.
- El score XGBoost redistribuye la prima dentro de la cartera sin alterar la suficiencia agregada.
- BonusMalus domina el feature importance del modelo de scoring. Su eliminación produce una degradación simétrica (~2%) tanto en GLM como en XGBoost, lo que sugiere que las variables restantes capturan información de riesgo similar.

---

## Estructura del Repositorio

```
actuarial-pricing-ml/
├── notebooks/
│   └── pricing_actuarial_ML_V5.ipynb   # Notebook principal
├── results/                             # Gráficos y tablas exportados
├── .gitignore
└── README.md
```

---

## Requisitos

```
scikit-learn
statsmodels
xgboost
pandas
numpy
matplotlib
seaborn
patsy
openml
```

Instalación:

```bash
pip install scikit-learn statsmodels xgboost pandas numpy matplotlib seaborn patsy openml
```

---

## Referencias

- Noll, A., Salzmann, R., Wüthrich, M.V. (2020). *Case Study: French Motor Third-Party Liability Claims*. SSRN. [https://ssrn.com/abstract=3164764](https://ssrn.com/abstract=3164764)
- Wüthrich, M.V., Merz, M. (2023). *Statistical Foundations of Actuarial Learning and its Applications*. Springer. [https://link.springer.com/book/10.1007/978-3-031-12409-9](https://link.springer.com/book/10.1007/978-3-031-12409-9)
- Dataset freMTPL2: [https://www.openml.org/d/41214](https://www.openml.org/d/41214)
