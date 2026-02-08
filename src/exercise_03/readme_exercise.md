
# Ejercicio 03 — Aprender una función sinusoidal con PyTorch

## Objetivo

Ajustar un modelo de regresión (MLP) para aproximar una función sinusoidal ruidosa generada sintéticamente. El objetivo es entrenar un modelo que prediga valores continuos y evaluar su capacidad de generalización en conjuntos de validación y test.

## Formalización

### Formalización (Inferencia)

Dado un escalar de entrada `x` en el rango [0, 100], el modelo debe predecir el valor correspondiente `y` según la función sinusoidal:

y = 100 * sin(8 * π * x / 100) + 2

En práctica, las observaciones contienen ruido aditivo gaussiano N(0, noise_std^2), por lo que el modelo aprende a aproximar la función subyacente a pesar del ruido.

### Formalización (Entrenamiento)

El objetivo de entrenamiento es encontrar los parámetros θ del modelo f_θ(x) que minimicen la función de pérdida MSE sobre el conjunto de datos de entrenamiento:

L(θ) = (1/N) * Σ (f_θ(x_i) - y_i)^2

donde N es el número de muestras en el conjunto de entrenamiento.

## Métricas de evaluación

- MSE (Mean Squared Error): métrica de pérdida principal usada durante entrenamiento.
- MAE (Mean Absolute Error): métrica complementaria, más robusta a outliers.
- R² (coeficiente de determinación): proporciona intuición de la fracción de varianza explicada.

Se guardan métricas por split (train/val/test) en `outs/exercise_03/metrics.png` y/o `metrics.csv` si el pipeline lo produce.

## Datos

### Descripción del dataset

- `x`: valores muestreados uniformemente en el rango [0, 100].
- `y`: función determinística 100*sin(8πx/100) + 2 más ruido gaussiano N(0, noise_std^2).
- Parámetros por defecto en el código: `noise_std=20`, tamaño total configurable.

### Preparación y preprocesamiento

- Se calcula la media (`x_mean`) y desviación estándar (`x_std`) únicamente sobre el conjunto de entrenamiento.
- Se aplica normalización estándar (z-score) a `x`: x_norm = (x - x_mean) / x_std.
- Los parámetros de normalización se persisten en `outs/exercise_03/norm_params.npz` para reutilizarlos en `evaluate.py`.
- No se normaliza `y` en la implementación por defecto; si la escala dificulta la convergencia, considerar normalizar `y` y desnormalizar las predicciones al graficar.

### Aumento de datos

- No se aplica aumento en este ejercicio (no es necesario para funciones sintéticas como la sinusoidal).

## Modelo

### Funciones de pérdida candidatas

**Candidatos principales:**

- **MSELoss (Mean Squared Error)**: penaliza errores grandes al elevar al cuadrado; ideal para regresión cuando queremos minimizar magnitud del error.
- **MAELoss (Mean Absolute Error)**: menos sensible a outliers que MSE; útil si hay ruido extremo.
- **HuberLoss**: compromiso entre MSE y MAE; robusto a outliers y suave alrededor de cero.

### Función de pérdida seleccionada

Se selecciona **MSELoss** porque:
- Coincide directamente con la función objetivo de regresión (minimizar varianza del error).
- Proporciona gradientes estables para optimización con Adam.
- Facilita la interpretación de la magnitud del error.

### Arquitecturas posibles

**Opciones exploradas:**

1. **MLP pequeña (1 capa):** `input(1) → hidden(8) → output(1)`. Riesgo de subajuste.
2. **MLP media (3 capas):** `input(1) → hidden(16) → hidden(16) → hidden(16) → output(1)`. **Seleccionada por defecto.**
3. **MLP profunda (5+ capas):** Más capacidad pero riesgo de overfitting.
4. **Variaciones:** cambiar `hidden_dim` a 32, 64 o añadir Dropout/BatchNorm.

### Activación de la capa de salida

La capa de salida **no tiene activación (identidad lineal)** porque:
- Para regresión continua necesitamos permitir cualquier rango de valores (no acotados).
- Las funciones sinusoidales pueden tomar valores negativos y positivos.
- ReLU saturarías a 0 para predicciones negativas; Tanh/Sigmoid limitaría el rango artificialmente.

### Otras consideraciones

- **Batch Normalization:** recordar desactivarla en `model.eval()` para evaluación.
- **Dropout:** puede mejorar generalizacion si hay overfitting; se desactiva en evaluación.
- **Inicialización de pesos:** PyTorch usa inicialización por defecto.
- **Versioning:** documentar los valores seleccionados para `hidden_dim` y arquitectura al guardar `state_dict`.
## Entrenamiento

### Hiperparámetros de entrenamiento

- Optimizer: `Adam`
- Learning rate: `0.001`
- Batch size: `10`
- Épocas: `100`
- Loss: `MSELoss`

### Gráfica de la función de pérdida

![image](../../outs/exercise_03/loss_plot.png)

### Discusión del proceso de entrenamiento

El proceso de entrenamiento sigue estos pasos:

1. **Normalización:** se calcula `x_mean` y `x_std` en el split de entrenamiento y se aplica z-score a todas las entradas.
2. **Entrenamiento:** el MLP se entrena minimizando MSE sobre el conjunto de entrenamiento con batches de tamaño 10.
3. **Validación:** después de cada época, se evalúa en el conjunto de validación (sin actualizar pesos) para monitoreo.
4. **Guardado:** se guarda el `state_dict` del mejor modelo observado por pérdida de validación en `best_model.pth`.
5. **Persistencia:** los parámetros de normalización se guardan en `norm_params.npz` para reutilizarlos en evaluación.

**Observaciones clave:**
- Calcular normalización **solo sobre el split de entrenamiento** es crítico para evitar data leakage.
- Si la pérdida no desciende: verificar learning rate, normalización, arquitectura o escala de `y`.
- Si hay overfitting severo: reducir capacidad, añadir regularización (weight decay, Dropout) o aumentar datos.

## Evaluación

### Métricas de evaluación

- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (coeficiente de determinación)

![image](../../outs/exercise_03/train_regression_plot.png)

![image](../../outs/exercise_03/validation_regression_plot.png)

![image](../../outs/exercise_03/test_regression_plot.png)

Metrics for each dataset is depicted: 

![image](../../outs/exercise_03/metrics.png)

### Resultados de evaluación

Here you have examples of evaluation results for train, validation and test sets.

Example for train set:

![image](../../outs/exercise_03/train_data_points_plot.png)


Example for validation set:

![image](../../outs/exercise_03/validation_data_points_plot.png)


Example for test set:

![image](../../outs/exercise_03/test_data_points_plot.png)


### Discusión de los resultados

**Interpretación de métricas:**

- **MSE bajo (< 100) + R² > 0.8:** el modelo captura bien la función sinusoidal.
- **MSE medio (100-400) + R² 0.5-0.8:** ajuste aceptable pero hay margen de mejora (posible subajuste).
- **MSE alto (> 400) + R² < 0.5:** ajuste pobre; revisar arquitectura, normalización, learning rate.

**Detección de problemas:**

- **Overfitting:** train MSE << val/test MSE. Solución: reducir capacidad, weight decay, Dropout.
- **Underfitting:** todos (train, val, test) con MSE alto. Solución: aumentar `hidden_dim`, más épocas, reduce regularización.
- **Data mismatch:** val y test difieren mucho. Revisión de normalización y splits.

**Esperado con `noise_std=20`:**
- El modelo capturará la forma general sinusoidal pero no los picos exactos.
- Ruido limita el MSE mínimo alcanzable (noise floor).

## Bucles de mejora (Feedback loop)

**Proceso iterativo de mejora:**

1. **Experimento base:** entrenar con configuración default y guardar métricas (MSE, MAE, R²) en train/val/test.
2. **Hipótesis:** si val MSE es alto → aumentar `hidden_dim`; si hay overfitting → añadir weight decay.
3. **Variar parámetro:** cambiar un hiperparámetro (p. ej. `hidden_dim=32`) y re-entrenar.
4. **Comparar:** graficar curvas de pérdida y comparar métricas finales con baseline.
5. **Decidir:** si mejora, aceptar y continuar; si empeora, revertir y probar otro parámetro.
6. **Registrar:** mantener tabla CSV con columnas: `experiment_id, hidden_dim, lr, batch_size, epochs, train_mse, val_mse, test_mse`.

**Parámetros a variar (por orden de impacto):**
- `hidden_dim`: 8, 16, 32, 64 (afecta capacidad del modelo).
- `lr`: 0.0001, 0.0005, 0.001, 0.01 (velocidad de convergencia).
- `batch_size`: 5, 10, 20, 32 (estabilidad del gradiente).
- `num_epochs`: 50, 100, 200, 500 (tiempo de entrenamiento).
- `weight_decay`: 0.0, 0.0001, 0.001 (regularización L2).

## Preguntas

Por favor responde las siguientes preguntas. Incluye gráficos si es necesario. Guarda los gráficos en la carpeta `outs/exercise_03`.

### ¿Cuáles son las diferencias que encontraste entre el modelo anterior y este?

En comparación con `exercise_02`, este modelo presenta una arquitectura con mayor capacidad (más capas y/o mayor `hidden_dim`) y un flujo de preprocesado más consistente: la normalización z‑score de las entradas se calcula exclusivamente sobre el split de entrenamiento y se persiste para evaluación. Además, los hiperparámetros por defecto (optimizer, learning rate, batch size y número de épocas) han sido ajustados para esta tarea más compleja. Estas diferencias suelen traducirse en cambios observables en las métricas (MSE, MAE, R²) y en los gráficos de ajuste: el modelo actual tiende a representar mejor la forma sinusoidal, mientras que el modelo anterior, con menos capacidad, mostraba un ajuste más rígido y posible subajuste.

### ¿Generaliza bien el modelo a datos nuevos?

La generalización se evalúa comparando las métricas en test y validación: si `test_mse` es aproximadamente igual a `val_mse` y ambos son razonablemente bajos, el modelo generaliza bien, teniendo en cuenta que existe un "noise floor" impuesto por `noise_std=20` que limita el MSE mínimo alcanzable. Si `test_mse` es significativamente mayor que `val_mse`, es indicativo de overfitting y conviene reducir capacidad o añadir regularización (weight decay, Dropout, early stopping). Si todos los MSE son elevados (train, val y test), hay indicios de underfitting y se deberían aumentar la capacidad del modelo, entrenar más épocas o revisar la normalización (por ejemplo normalizar también `y` y desnormalizar las predicciones para la interpretación).






