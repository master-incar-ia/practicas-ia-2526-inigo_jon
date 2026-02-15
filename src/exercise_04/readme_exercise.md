
# Ejercicio 4: Crear un Modelo de Deep Learning para clasificación de imágenes en PyTorch con dataset CIFAR-10

## Objetivo

Desarrollar un modelo de red neuronal convolucional (CNN) que clasifique imágenes del conjunto de datos CIFAR-10. El modelo debe evaluar su rendimiento calculando métricas de clasificación e incluir una matriz de confusión. Se comparan los resultados con el ejercicio anterior (ejercicio 3) para analizar el efecto de la augmentación de datos.

## Formalización de la Tarea

### Formalización de la Tarea (Inferencia)

Dada una imagen de entrada de 32x32 píxeles en RGB, clasificarla en una de las 10 clases de CIFAR-10 (avión, automóvil, pájaro, gato, ciervo, perro, rana, caballo, barco, camión). El modelo debe predecir la probabilidad de pertenencia a cada clase y seleccionar la clase con mayor probabilidad.

### Formalización de la Tarea (Entrenamiento)

Entrenar una CNN con datos etiquetados del conjunto de entrenamiento de CIFAR-10 (50,000 imágenes) mediante optimización de gradiente descendente. El objetivo es minimizar la función de pérdida CrossEntropyLoss, validando continuamente con el conjunto de validación (15% de los datos de entrenamiento) y guardando el mejor modelo según la pérdida de validación.

## Métricas de Evaluación

Las métricas utilizadas para evaluar el rendimiento del modelo son:
- **Exactitud (Accuracy)**: Porcentaje de predicciones correctas
- **Precisión (Precision)**: Proporción de predicciones positivas correctas (promedio ponderado)
- **Exhaustividad (Recall)**: Proporción de instancias positivas correctamente identificadas (promedio ponderado)
- **F1-Score**: Media armónica entre precisión y exhaustividad (promedio ponderado)

## Consideraciones sobre los Datos

### Descripción del dataset

El dataset CIFAR-10 contiene 60,000 imágenes en color de 32x32 píxeles distribuidas en 10 clases con 6,000 imágenes por clase:
- **Clases**: avión, automóvil, pájaro, gato, ciervo, perro, rana, caballo, barco, camión
- **Conjunto de entrenamiento**: 50,000 imágenes
- **Conjunto de prueba**: 10,000 imágenes
- **Tamaño de imagen**: 32x32 píxeles, 3 canales (RGB)

Los datos se dividieron en:
- 85% para entrenamiento: 42,500 imágenes
- 15% para validación: 7,500 imágenes
- 100% del conjunto original para prueba: 10,000 imágenes
Consideraciones del Modelo


### Función de Pérdida Seleccionada

Se seleccionó **CrossEntropyLoss** por ser la opción estándar para clasificación multiclase balanceada. Esta función calcula automáticamente el softmax de las logits del modelo antes de calcular la pérdida, lo que asegura probabilidades válidas.

### Arquitecturas Posibles

Se consideraron varias arquitecturas:
- **Red neuronal completamente conectada (MLP)**: Demasiados parámetros para imágenes 32x32, propenso a computación excesiva
- **CNN simple**: Extrae características locales eficientemente, es adecuada para imágenes pequeñas
- **VGG-like**: Múltiples capas pequeñas y secuenciales (potencialmente mejor pero más parámetros)
- **ResNet**: Conexiones de salto para entrenamientos más profundos (excesivo para este dataset)

Se eligió una **CNN simple y eficiente** con 3 capas convolucionales seguidas de 3 capas densas.

### Arquitectura Seleccionada (SimpleCNN)

**Capas convolucionales**:
- Conv2D(3→32, kernel=3, padding=1) + ReLU + MaxPool(2x2)
- Conv2D(32→64, kernel=3, padding=1) + ReLU + MaxPool(2x2)
- Conv2D(64→128, kernel=3, padding=1) + ReLU + MaxPool(2x2)
- Flattening: 128 × 4 × 4 = 2,048 características

**Capas densas**:
- FC(2048→256) + ReLU
- FC(256→128) + ReLU
- FC(128→10) (salida multiclase, sin activación en la última capa)

### Activación de la Última Capa

No hay activación explícita en la última capa. Se devuelven los **logits** directamente, ya que CrossEntropyLoss espera logits sin aplicar softmax previamente.

### Hiperparámetros de Entrenamiento

| Hiperparámetro | Valor |
|---|---|
| Algoritmo de optimización | Adam |
| Tasa de aprendizaje inicial | 0.001 |
| Tamaño de batch | 64 |
| Número de épocas | 100 |
| Función de pérdida | CrossEntropyLoss |
| Planificador de tasa de aprendizaje | Ninguno (fijo) |
| Estrategia de regularización | Dropout (comentado) |

### Gráfico de la función de pérdida

![image](ción

### Métricas de Evaluación

Las métricas obtenidas en los tres conjuntos de datos son:

| Métrica | Entrenamiento | Validación | Prueba |
|---|---|---|---|
| **Exactitud (Accuracy)** | 89.76% | 89.89% | 82.15% |
| **Precisión (Precision)** | 89.72% | 89.91% | 82.04% |
| **Exhaustividad (Recall)** | 89.76% | 89.89% | 82.15% |
| **F1-Score** | 89.68% | 89.84% | 81.98% |

![image](../../outs/exercise_04/metrics.png)

### Resultados de la Evaluación

**Matriz de confusión en conjunto de entrenamiento:**
![image](../../outs/exercise_04/train_confusion_matrix.png)

**Matriz de confusión en conjunto de validación:**
![image](../../outs/exercise_04/validation_confusion_matrix.png)

**Matriz de confusión en conjunto de prueba:**
![image](../../outs/exercise_04/test_confusion_matrix.png)

### Discusión de los Resultados

**¿Cómo el modelo resuelve el problema?**

El modelo SimpleCNN extrae características espaciales mediante capas convolucionales que detectan patrones de bajo nivel (bordes, texturas) y posteriormente patrones más complejos (formas, objetos). Las capas densas funcionan como un clasificador que mapea estas características a las 10 clases. La arquitectura se especializa bien en imágenes pequeñas de 32x32 píxeles.

**¿Existe overfitting, underfitting u otros problemas?**

Existe un **gap de rendimiento entre validación y prueba** (89.89% vs 82.15%), lo que indica:
- El modelo se ajusta bien al conjunto de validación
- El conjunto de prueba presenta cierta distribución diferente
- Hay ligera evidencia de overfitting, pero el modelo generaliza razonablemente

No hay evidencia de underfitting (ambos conjuntos de entrenamiento y validación tienen rendimiento alto).

**¿Cómo podemos mejorar el modelo?**

1. **Mejor aprovechamiento de los datos**: Más transformaciones (rotación, zoom, cambios de brillo)
2. **Usar dropout efectivamente**: Descomentar dropout en capas densas para regularización
3. **Arquitectura más profunda**: Añadir más capas convolucionales o usar arquitecturas pre-entrenadas
4. **Técnicas de regularización**: L1/L2, batch normalization
5. **Fine-tuning**: Usar modelos pre-entrenados (ResNet, VGG) con transfer learning

**¿Cómo generalizará el modelo a nuevos datos?**

El modelo generaliza razonablemente bien con un 82% de exactitud en el conjunto de prueba. Para datos nuevos que sigan la distribución de CIFAR-10, se espera rendimiento similar. Sin embargo, para imágenes muy diferentes (diferentes ángulos, condiciones de iluminación extremas, objetos parcialmente ocultos), el rendimiento puede degradarse.

**Proceso de mejora del modelo:**

1. **Primera iteración**: Modelo inicial SimpleCNN con arquitectura de 3 capas conv + 3 capas densas, sin regularización activa
2. **Validación inicial**: Se observó buen rendimiento en entrenamiento y validación (89.8%) pero degradación en prueba (82%)
3. **Análisis**: El gap validación-prueba sugiere ligero overfitting y posible falta de generalización
4. **Acciones consideradas** (no implementadas, pero evidenciadas por el código):
   - Dropout disponible pero comentado (líneas en model.py)
   - Augmentación de datos ya implementada (RandomHorizontalFlip, RandomCrop)
   - Arquitectura validada como suficientemente compleja

El modelo actual representa un punto de equilibrio razonable entre complejidad y rendimiento, con potencial de mejora mediante regularización más agresiva.

| Iteración | Configuración | Exactitud (Prueba) | Observaciones |
|---|---|---|---|
| 1 | SimpleCNN base, sin dropout activo | 82.15% | Buen rendimiento, gap V-T de 7.7% |

## Preguntas

### ¿Cuáles son las diferencias encontradas entre el modelo anterior (Ejercicio 3) y este (Ejercicio 4)?

El ejercicio anterior (Ejercicio 3) utilizaba una red neuronal completamente conectada (MLP/Dense) para clasificación, mientras que este ejercicio (Ejercicio 4) utiliza una CNN con capas convolucionales. Las principales diferencias son:

1. **Tipo de arquitectura**: 
   - Ej. 3: MLP - conexiones completas entre todas las neuronas
   - Ej. 4: CNN - capas convolucionales que procesan localidad espacial

2. **Extracción de características**:
   - Ej. 3: Sin entendimiento de estructura espacial
   - Ej. 4: Extrae patrones locales (bordes, texturas) en primeras capas

3. **Número de parámetros**:
   - Ej. 3: Más parámetros debido a conectividad completa
   - Ej. 4: Menos parámetros gracias a compartición de pesos convolucionales

4. **Eficacia en imágenes pequeñas**:
   - Ej. 3: Menos eficiente para 32x32
   - Ej. 4: Diseñado específicamente para imágenes pequeñas

5. **Augmentación de datos**:
   - Ej. 3: Posiblemente sin augmentación o minimal
   - Ej. 4: Con augmentación (flip, crop) para mejorar robustez

### ¿Generaliza bien el modelo a nuevos datos?

**Respuesta: Moderadamente sí, con consideraciones.**

**Evidencia de buena generalización**:
- Exactitud en prueba: 82.15% (razonablemente cercana a validación del 89.89%)
- Métricas balanceadas (Precisión, Recall, F1 similares)
- Convergencia estable sin signos de inestabilidad

**Limitaciones de generalización**:
- Gap validación-prueba de 7.7% sugiere distribución ligeramente diferente
- Posible overfitting ligero al conjunto de validación
- Imágenes nuevas muy diferentes de CIFAR-10 podrían tener peor rendimiento

**Conclusión**: El modelo generaliza bien a datos de la misma distribución que CIFAR-10, pero muestra cierto grado de sobreespecialización. La aplicación a dominios muy diferentes (imágenes médicas, satélite, arte) probablemente requeriría fine-tuning o transfer learning._plot.png)