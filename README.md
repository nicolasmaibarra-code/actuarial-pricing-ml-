# Actuarial Pricing with Machine Learning
### GLM vs ML — Motor Insurance Frequency-Severity Modeling

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Domain](https://img.shields.io/badge/Domain-Insurance%20Pricing-orange)
![Models](https://img.shields.io/badge/Models-GLM%20%7C%20RF%20%7C%20XGBoost-purple)

---

## ¿De qué trata este proyecto?

Las aseguradoras calculan cuánto cobrarle a cada cliente usando **modelos actuariales clásicos** que tienen décadas de uso regulatorio. Me propuse a investigar si es posible obtener mejores resultados mediante la utilización de modelos de aprendizaje automatico.

```
Datos reales de siniestros → Modelado de frecuencia y severidad →
Comparativa GLM vs ML → Enfoques híbridos → Prima pura por póliza
```

---

## Dataset

**freMTPL2** — Motor Third-Party Liability, mercado francés  
Fuente: OpenML (IDs 41214 y 41215)

| Dataset | Filas | Descripción |
|---|---|---|
| freMTPL2freq | ~678.000 | Una fila por póliza: exposición, características del vehículo y conductor, número de siniestros |
| freMTPL2sev | ~26.000 | Una fila por siniestro: importe reclamado |

Variables utilizadas: `VehAge`, `DrivAge`, `BonusMalus`, `VehPower`, `Density`, `Area`, `VehBrand`, `VehGas`, `Region`, `Exposure`

---

## Estructura del proyecto

```
├── pricing_actuarial_ML.ipynb   # Notebook principal
└── README.md
```

### Secciones del notebook

| Sección | Contenido |
|---|---|
| 0. Setup | Imports, constantes globales, configuración |
| 1. Introducción | Marco teórico: frecuencia × severidad = prima pura |
| 2. Preparación de datos | Carga, merge, encoding de categóricas, train/test split |
| 3. Modelos de Frecuencia | GLM Poisson · Random Forest · XGBoost |
| 4. Comparativa Frecuencia | Métricas, distribuciones, feature importance |
| 5. Modelos de Severidad | GLM Gamma · XGBoost |
| 6. Prima Pura | Frecuencia × Severidad para GLM y XGBoost |
| 7. Conclusiones parciales | Por qué el GLM gana en este contexto |
| 8. Two-Stage GLM + ML | XGBoost sobre residuos del GLM |
| 9. Scoring híbrido | Prima GLM × score de riesgo relativo ML |

---

## Decisiones técnicas clave

### ¿Por qué GLM Poisson para frecuencia?
La variable objetivo (`ClaimNb`) es un conteo discreto no negativo. La familia Poisson con link log es la especificación estadísticamente correcta para este tipo de variable. El offset `log(Exposure)` escala el número esperado de siniestros al tiempo asegurado de cada póliza 

### ¿Por qué GLM Gamma para severidad?
`ClaimAmount` es continuo, positivo y con cola derecha pronunciada — exactamente la forma que describe la distribución Gamma. Modelar esto con MSE (como hace XGBoost por defecto) penaliza outliers de forma inapropiada para el contexto actuarial.

### ¿Por qué ML no supera al GLM aquí?
Tres razones estructurales:
1. **Sparsidad**: el 93% de las pólizas tiene `ClaimNb = 0`. Con tan poca señal positiva, los árboles no encuentran splits informativos.
2. **BonusMalus**: esta variable es un resumen actuarial acumulado de toda la historia del conductor — ya contiene implícitamente la información que otras features aportarían.
3. **Especificación correcta**: cuando el proceso generador de datos es realmente Poisson, el GLM captura toda la señal disponible. El residuo es ruido, no patrón.

### ¿Cuándo agrega valor el ML entonces?
En los enfoques híbridos de las Secciones 8 y 9: como capa de corrección sobre el GLM (Two-Stage) o como score de riesgo relativo que preserva la suficiencia de prima de cartera (Scoring). 

---

## Resultados

| Modelo | RMSE | MAE | R² |
|---|---|---|---|
| GLM Poisson | — | — | — |
| Random Forest | — | — | — |
| XGBoost | — | — | — |
| Two-Stage (GLM + XGB) | — | — | — |
| Scored (GLM × score XGB) | — | — | — |

> Los valores exactos varían con la versión del dataset. Ejecutar el notebook para ver los resultados actualizados.

---

## Conclusión principal

> El GLM Poisson-Gamma sigue siendo el estándar de la industria por razones bien fundadas: especificación estadística correcta, interpretabilidad regulatoria y comportamiento robusto con datos sparse. Los modelos de ML agregan valor no como reemplazo sino como capa de inteligencia comercial encima del GLM. Esta es exactamente la arquitectura que separa a las insurtechs maduras de los proyectos académicos.

---

## Referencias

- Wüthrich, M. V. & Merz, M. (2023). *Statistical Foundations of Actuarial Learning and its Applications*. Springer. [Disponible en SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3822407)
- Noll, A., Salzmann, R. & Wüthrich, M. V. (2020). *Case Study: French Motor Third-Party Liability Claims*. SSRN.

---

## Tecnologías

`Python` · `pandas` · `numpy` · `statsmodels` · `scikit-learn` · `XGBoost` · `matplotlib` · `seaborn`

---

## Autor

Proyecto de pricing actuarial desarrollado como parte de mi portfolio profesional en el cruce entre actuaría tradicional y Machine Learning aplicado a seguros.
