# Simple 2D Anomaly Detection Tutorial - Evaluation Questionnaire

## Student Information
- **Name**: Iñigo Martínez Peña y Jon Garzón García
- **Date**: 23/03/2026
- **Course**: Applications of AI for Industrial Control
- **Tutorial**: Simple 2D Gaussian Anomaly Detection

---

## Section 1: Theoretical Understanding (25 points)

### 1.1 Gaussian Model Fundamentals (15 points)

**Question 1** (5 points): What are the main parameters that define a Gaussian multivariate distribution for 2D data?
- a) Only the mean vector
- b) Only the covariance matrix
- c) Mean vector and covariance matrix
- d) Standard deviation and correlation coefficient

La c

**Question 2** (5 points): In the tutorial dataset, what were the three types of anomalies generated?
- a) High temperature, Low pressure, Medium values
- b) High temperature, High pressure, Low temperature-pressure
- c) Overheating, Undercooling, Normal operation
- d) Temperature spikes, Pressure drops, Random noise

La b

**Question 3** (5 points): True/False: The Gaussian anomaly detector requires labeled anomaly data during training.

Answer: Falso

Explanation: El detector Gaussiano suele entrenarse de forma no supervisada con datos “normales”: estima 𝜇 y Σ del comportamiento normal y marca como anomalías puntos con baja probabilidad bajo ese modelo. No necesita etiquetas de anomalía para entrenar.

### 1.2 Anomaly Scoring (10 points)

**Question 4** (5 points): What does a higher anomaly score indicate?
- a) Higher probability of being normal
- b) Lower probability of being normal
- c) Data point is closer to the mean
- d) Data point has higher variance

La b

**Question 5** (5 points): Fill in the blank: The anomaly score is calculated as the _________________ of the probability density, which means _________________ probability results in a _________________ anomaly score.

Answer: "negative log" and "lower" and "higher"

## Section 2: Visualization and Interpretation (25 points)

### 2.1 2D Visualization (25 points)

**Question 6** (10 points): Explain what probability contours represent in the 2D visualization and how they help in understanding the anomaly detection model.

_Your answer:_
Los contornos de probabilidad representan curvas donde la densidad 𝑝(𝑥) del modelo Gaussiano es constante (misma “altura” de probabilidad). En 2D suelen verse como elipses (según la covarianza). Ayudan a entender el modelo porque muestran dónde está la región normal (alta probabilidad, cerca del centro) y cómo se “estira” la distribución según la correlación/varianza. Los puntos que caen en zonas de contornos muy bajos (muy lejos del centro o en direcciones poco probables) son candidatos a anomalía. Además, permiten visualizar cómo un umbral corta la distribución y define la frontera entre normal y anómalo.

**Question 7** (8 points): In the tutorial, what happened to the decision boundary when the threshold percentile was increased from 85% to 95%?
- a) The boundary became more restrictive (smaller normal region)
- b) The boundary became less restrictive (larger normal region)
- c) The boundary shape changed but size remained the same
- d) No change occurred in the boundary

La b

**Question 8** (7 points): Describe the difference between True Positives, False Positives, True Negatives, and False Negatives in the context of anomaly detection.

- True Positives: anomalías reales detectadas como anomalías.
- False Positives: datos normales marcados erróneamente como anomalías (falsas alarmas).
- True Negatives: datos normales clasificados como normales.
- False Negatives: anomalías reales no detectadas (clasificadas como normales).

---

## Section 3: Performance Metrics (25 points)

### 3.1 Metrics Calculation (25 points)

**Question 9** (8 points): What does the F1-score represent in anomaly detection?
- a) The accuracy of the model
- b) The harmonic mean of precision and recall
- c) The area under the ROC curve
- d) The correlation between features

La b

**Question 10** (9 points): Given the following confusion matrix for an anomaly detection model:

```
              Predicted
              Normal  Anomaly
Actual Normal   450      50
       Anomaly   15      35
```

Calculate:
- Precision: 0.4118
- Recall: 0.7
- F1-Score: 0.5185

Show your calculations:

"Precision" = TP / (TP + FP) = 35 / (35 + 50) = 35 / 85 = 0.4118

"Recall" = TP / (TP + FN) = 35 / (35 + 15) = 35 / 50 = 0.7

"F1-Score" = 2 × (Precision × Recall) / (Precision + Recall) = 2 × (0.4118 × 0.7) / (0.4118 + 0.7) = 2 × 0.2883 / 1.1118 = 0.5761 / 1.1118 = 0.5185

**Question 11** (8 points): What is the trade-off when selecting threshold percentiles?
- a) High percentile: More false positives, fewer false negatives
- b) High percentile: Fewer false positives, more false negatives
- c) Low percentile: Fewer false positives, more false negatives
- d) Percentile has no effect on false positives/negatives

La b

---

## Section 4: Industrial Applications (15 points)

### 4.1 Real-World Applications (15 points)

**Question 12** (8 points): List three real-world industrial applications where 2D Gaussian anomaly detection could be effectively used. For each application, specify the two variables that would be monitored.

Applications:
1. Mantenimiento predictivo de motores: temperatura del motor y vibración RMS.
2. Presión de línea y temperatura de descarga.
3. Temperatura del reactor y presión del reactor.

**Question 13** (7 points): Which of the following is NOT a limitation of the Gaussian multivariate method?
- a) Assumes data follows Gaussian distribution
- b) Cannot handle non-linear anomaly patterns
- c) Requires large amounts of labeled training data
- d) Sensitive to outliers in training data

La c

---

## Section 5: Critical Thinking (10 points)

### 5.1 Implementation Strategy (10 points)

**Question 14** (10 points): You are implementing anomaly detection for a manufacturing process that monitors motor temperature and vibration. Based on the tutorial concepts:

1. How would you collect and prepare training data?
2. What threshold selection strategy would you use and why?
3. How would you handle the trade-off between false alarms and missed anomalies?
4. What additional considerations would you have for a real industrial deployment?

_Your answer:_
Para detectar anomalías con temperatura del motor y vibración, primero recogería datos que representen bien la operación normal (distintas cargas, turnos y condiciones). Luego limpiaría el dataset (faltantes y lecturas raras), sincronizaría ambas señales y, si la vibración viene como señal bruta, la resumiría con una medida estable (por ejemplo RMS). Después aplicaría z-score por variable usando media y desviación calculadas en train, y separaría train/validación/test mejor de forma temporal para no mezclar información.

El umbral lo elegiría con un percentil del anomaly score en validación: por ejemplo 95 o 99. Es fácil de controlar y explicar. Si tuviera algunos fallos confirmados, ajustaría ese percentil para mejorar la métrica que nos interese (por ejemplo F1 o recall).

El compromiso principal es entre falsas alarmas y anomalías no detectadas. Si es crítico no perder fallos, bajo el umbral (más sensibilidad) y acepto más alertas. Si las falsas alarmas son muy costosas, subo el umbral (menos alertas) y acepto que alguna anomalía se escape. En industria suele funcionar bien tener dos niveles: warning y alarm.

Para desplegarlo en fábrica, además tendría en cuenta: cambios con el tiempo (deriva, mantenimiento), control de calidad de sensores, registro y trazabilidad de alertas, integración con el sistema de planta y un plan claro de actuación cuando salta una alarma. También vigilaría que el entrenamiento no incluya outliers “normales” que deformen el modelo, y si la forma de los datos no se parece a una nube gaussiana, consideraría un método más flexible.
---

## Scoring Summary

| Section | Points Earned | Total Points |
|---------|---------------|--------------|
| 1. Theoretical Understanding | _____ / 25 | 25 |
| 2. Visualization and Interpretation | _____ / 25 | 25 |
| 3. Performance Metrics | _____ / 25 | 25 |
| 4. Industrial Applications | _____ / 15 | 15 |
| 5. Critical Thinking | _____ / 10 | 10 |
| **Total** | **_____ / 100** | **100** |

**Grade**: ____________

**Comments**:
_______________________________________________________________________________
_______________________________________________________________________________
