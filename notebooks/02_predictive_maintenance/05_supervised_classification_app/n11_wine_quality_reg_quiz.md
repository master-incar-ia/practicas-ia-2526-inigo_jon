# Wine Quality Regression Tutorial - Evaluation Questionnaire

## Student Information
- **Name**: Iñigo Martínez y Jon Garzón
- **Date**: 10/03/2026
- **Course**: Applications of AI for Industrial Control
- **Tutorial**: Wine Quality Regression for Continuous Quality Assessment

---

## Section 1: Theoretical Understanding (40 points)

### 1.1 Regression vs Classification (20 points)

**Question 1** (5 points): Why is regression better than multiclass classification for wine quality prediction in industrial quality control?
- a) Regression models are faster to train
- b) Regression provides continuous, precise quality scores rather than discrete categories, enabling more nuanced quality assessment
- c) Classification models require more data
- d) Regression uses less computational resources

La b.

**Question 2** (5 points): What is the primary difference in output between classification and regression for wine quality?
- a) Classification: 3.2, 5.7, 6.8; Regression: 3, 5, 6, 7
- b) Classification: 3, 4, 5, 6, 7, 8, 9; Regression: 3.2, 5.7, 6.8, 7.3
- c) Both produce the same output format
- d) Classification uses probabilities, regression uses integers

La b.

**Question 3** (5 points): Which loss function is most appropriate for the wine quality regression task?
- a) Cross-Entropy Loss
- b) Mean Squared Error (MSE)
- c) Binary Cross-Entropy
- d) Hinge Loss

La b.

**Question 4** (5 points): Which activation function is used in the output layer of the regression neural network?
- a) ReLU
- b) Sigmoid
- c) Softmax
- d) Linear (no activation)

La d.

### 1.2 Model Architecture and Training (20 points)

**Question 5** (5 points): What is the purpose of scaling both features (X) and target (y) in the regression model?
- a) To make training faster only
- b) To facilitate neural network convergence and improve gradient flow for both inputs and outputs
- c) To reduce memory usage
- d) To prevent overfitting only

La b.

**Question 6** (5 points): The WineQualityRegressor includes dropout layers for:
- a) Faster training
- b) Uncertainty quantification and overfitting prevention
- c) Better gradient flow
- d) Memory optimization

La b.

**Question 7** (5 points): What is the purpose of the `predict_with_uncertainty()` method in the regression model?
- a) To make predictions faster
- b) To provide confidence intervals and uncertainty estimates for predictions
- c) To improve model accuracy
- d) To reduce computational cost

La b.

**Question 8** (5 points): What does the difference between MSE and MAE as regression metrics represent?
- a) MSE and MAE are identical
- b) MSE penalizes large errors more heavily (squared), MAE treats all errors equally (absolute)
- c) MSE is for classification, MAE is for regression
- d) MAE is always larger than MSE

La b.

---

## Section 2: Model Evaluation (30 points)

### 2.1 Regression Metrics (15 points)

**Question 9** (5 points): Which metric is most appropriate for evaluating the overall performance of a regression model?
- a) Accuracy
- b) F1-Score
- c) R² (coefficient of determination)
- d) Precision

La a.

**Question 10** (5 points): What does an R² score of 0.85 indicate?
- a) 85% of data points are correctly classified
- b) 85% of the variance in quality scores is explained by the model
- c) The model has 15% error rate
- d) 85% accuracy in predictions

La b.

**Question 11** (5 points): The residual analysis in regression helps to:
- a) Speed up training
- b) Identify prediction patterns and model bias
- c) Reduce model complexity
- d) Increase accuracy

La b.

### 2.2 Industrial Applications (15 points)

**Question 12** (15 points): In the context of continuous quality scoring, what advantage does regression provide for industrial applications? Provide specific examples of how precise quality measurements benefit manufacturing processes.

_Your answer:_
La regresión permite obtener valores continuos de calidad en lugar de categorías discretas, lo que proporciona una evaluación más precisa. En un entorno industrial esto permite detectar pequeñas diferencias de calidad entre lotes de vino. Por ejemplo, un vino con calidad 6.8 puede considerarse mejor que uno con 6.1 aunque ambos estén en la misma categoría si se usara clasificación. Esto ayuda a ajustar procesos de producción, seleccionar lotes para distintos mercados o decidir mezclas entre vinos para alcanzar una calidad objetivo. Además, la regresión puede identificar tendencias a lo largo del tiempo, como mejoras o deterioros en la calidad, lo que es crucial para el control de calidad continuo y la optimización de procesos en la industria vinícola.

---

## Section 3: Practical Implementation (20 points)

### 3.1 Uncertainty Quantification (10 points)

**Question 13** (10 points): Describe the uncertainty quantification approach used in the wine quality regressor. How does the model provide confidence intervals for its predictions and why is this important for industrial applications?

_Your answer:_
El modelo utiliza dropout durante la predicción para generar varias predicciones del mismo vino. Al repetir la predicción varias veces se obtiene una distribución de resultados. A partir de esa distribución se calcula una media (predicción final) y una desviación estándar, que permite construir intervalos de confianza.

Esto es importante en aplicaciones industriales porque permite saber cuándo el modelo está seguro y cuándo no. Si la incertidumbre es alta, el lote puede revisarse manualmente o analizarse con métodos adicionales. Esto ayuda a evitar decisiones erróneas basadas en predicciones poco confiables, lo que es crucial para mantener la calidad del producto y la satisfacción del cliente.

### 3.2 Loss Function Selection (10 points)

**Question 14** (10 points): Compare the advantages and disadvantages of using MSE vs MAE as loss functions for wine quality regression. When would you choose one over the other in an industrial setting?

_Your answer:_
El MSE penaliza más los errores grandes porque eleva el error al cuadrado. Esto hace que el modelo se centre en evitar predicciones muy alejadas del valor real, pero también lo hace más sensible a valores atípicos.

El MAE utiliza el valor absoluto del error, por lo que trata todos los errores de forma más uniforme y es más robusto frente a outliers.

En un entorno industrial se usaría MSE cuando los errores grandes son especialmente problemáticos y deben evitarse. En cambio, se preferiría MAE cuando el dataset tiene ruido o valores atípicos y se busca un modelo más robusto. Por ejemplo, si un lote de vino tiene una calidad extremadamente baja debido a un error de producción, el MSE penalizaría mucho esa predicción, mientras que el MAE permitiría que el modelo aprenda sin ser tan afectado por ese outlier.

---

## Section 4: Critical Thinking (10 points)

### 4.1 Industrial Implementation Strategy (10 points)

**Question 15** (10 points): You're deploying this regression model in a winery that produces 10,000 bottles per day. The quality team wants to use your model to optimize blending decisions (mixing different batches to achieve target quality scores). Design a strategy that uses the regression model's continuous predictions and uncertainty estimates to make optimal blending recommendations.

_Your answer:_
Se puede usar el modelo para predecir la calidad continua de cada lote antes del embotellado. Con esa predicción y su incertidumbre, cada lote recibe un valor estimado de calidad.

Después se pueden simular mezclas entre lotes para alcanzar una calidad objetivo. Por ejemplo, un lote con calidad 5.8 puede mezclarse con otro de 7.2 para obtener una mezcla cercana a 6.5. El modelo permite estimar la calidad esperada de cada combinación antes de realizarla.

La incertidumbre se usa para tomar decisiones más seguras. Si un lote tiene alta incertidumbre en la predicción, se puede enviar a análisis adicional o a cata humana antes de usarlo en una mezcla. Si la incertidumbre es baja, el sistema puede recomendar automáticamente la proporción de mezcla entre lotes para alcanzar la calidad objetivo de producción. Esto ayuda a mantener una calidad consistente y optimizar el uso de los lotes disponibles.

---

## Scoring Summary

| Section | Points Earned | Total Points |
|---------|---------------|--------------|
| 1. Theoretical Understanding | _____ / 40 | 40 |
| 2. Model Evaluation | _____ / 30 | 30 |
| 3. Practical Implementation | _____ / 20 | 20 |
| 4. Critical Thinking | _____ / 10 | 10 |
| **Total** | **_____ / 100** | **100** |

**Grade**: ____________

**Comments**:
_______________________________________________________________________________
_______________________________________________________________________________
