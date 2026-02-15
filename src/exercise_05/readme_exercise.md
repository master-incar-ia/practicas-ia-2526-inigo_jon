
# Ejercicio 5: Modelo de Deep Learning con Red Neuronal Completamente Conectada para CIFAR-10

## Objetivo

Desarrollar un modelo de red neuronal completamente conectada (MLP) para clasificación de imágenes en CIFAR-10. Evaluar el rendimiento del modelo calculando métricas de clasificación e incluyendo una matriz de confusión. Analizar comparativamente con ejercicios anteriores, especialmente el efecto de la augmentación de datos cuando se usa una arquitectura diferente (MLP vs CNN).

## Formalización de la Tarea

### Formalización de la Tarea (Inferencia)

Dada una imagen de entrada de 32x32 píxeles en RGB, aplanarla a un vector de 3072 características (3 canales × 32 × 32) y clasificarla en una de las 10 clases de CIFAR-10 mediante una red completamente conectada. El modelo predice probabilidades para cada clase y selecciona la clase con mayor probabilidad.

### Formalización de la Tarea (Entrenamiento)

Entrenar una MLP con datos etiquetados del conjunto de entrenamiento de CIFAR-10 (50,000 imágenes) mediante optimización Adam. El objetivo es minimizar la función de pérdida CrossEntropyLoss, validando continuamente con el conjunto de validación (15% de los datos de entrenamiento) y guardando el mejor modelo según la pérdida de validación.

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

### Funciones de Pérdida Adecuadas

Para un problema de clasificación multiclase como CIFAR-10, las opciones son:
- **CrossEntropyLoss**: Estándar para clasificación multiclase
- **Focal Loss**: Cuando hay desbalance de clases
- **Label Smoothing**: Para reducir sobrefitting

### Función de Pérdida Seleccionada

Se seleccionó **CrossEntropyLoss** como función estándar para clasificación multiclase en PyTorch.

### Arquitecturas Posibles

Se consideraron dos enfoques principales:
- **Red neuronal completamente conectada (MLP)**: Simple pero menos eficiente para imágenes, muchos parámetros
- **Red convolucional (CNN)**: Extrae características espaciales, más eficiente con menos parámetros

Este ejercicio construye una **MLP** para demostrar las limitaciones de las arquitecturas completamente conectadas en visión por computadora.

### Arquitectura Seleccionada (MLPClassifier)

**Arquitectura**:
- Entrada: 3072 características (aplanadas de 3×32×32)
- Capas ocultas: 512 → 256 → 128
- Salida: 10 neuronas (clasificación multiclase)

**Detalles**:
- Activación: ReLU en todas las capas ocultas
- Dropout: Disponible pero comentado (0.5)
- Capas: 4 capas lineales totales (3 ocultas + 1 salida)
- Parámetros totales: ≈ 1.9M

### Activación de la Última Capa

No hay activación explícita en la última capa. Se devuelven los **logits** directamente, ya que CrossEntropyLoss espera logits sin aplicar softmax previamente.

### Otras Consideraciones

- **Arquitectura completamente conectada**: Ignora la estructura espacial de las imágenes
- **Alto número de parámetros**: Propicio a overfitting
- **Optimizador**: Adam con lr=0.001
- *Entrenamiento

### Hiperparámetros de Entrenamiento

| Hiperparámetro | Valor |
|---|---|
| Algoritmo de optimización | Adam |
| Tasa de aprendizaje inicial | 0.001 |
| Tamaño de batch | 64 |
| Número de épocas | 100 |
| Función de pérdida | CrossEntropyLoss |
| Capas ocultas | [512, 256, 128] |
| Planificador de tasa de aprendizaje | Ninguno (fijo) |
| Estrategia de regularización | Dropout (comentado) |

### Gráfico de la función de pérdida
ción

### Métricas de Evaluación

Las métricas obtenidas en los tres conjuntos de datos son:

| Métrica | Entrenamiento | Validación | Prueba |
|---|---|---|---|
| **Exactitud (Accuracy)** | 54.96% | 55.52% | 51.67% |
| **Precisión (Precision)** | 56.46% | 57.38% | 52.80% |
| **Exhaustividad (Recall)** | 54.96% | 55.52% | 51.67% |
| **F1-Score** | 54.25% | 55.02% | 51.00% |

![image](../../outs/exercise_05/metrics.png)

### Resultados de la Evaluación

**Matriz de confusión en conjunto de entrenamiento:**
![image](../../outs/exercise_05/train_confusion_matrix.png)

**Matriz de confusión en conjunto de validación:**
![image](../../outs/exercise_05/validation_confusion_matrix.png)

**Matriz de confusión en conjunto de prueba:**
![image](../../outs/exercise_05/test_confusion_matrix.png)

### Discusión de los Resultados

**¿Cómo el modelo resuelve el problema?**

La MLP intenta aprender a clasificar imágenes aplanas tratando cada píxel como independiente. Sin embargo, al aplanar la imagen de 32×32 a 3072 características, se pierden las relaciones espaciales entre píxeles. La red aprende correlaciones globales débiles entre píxeles dispersos, lo que resulta en predicciones pobres (apenas mejor que aleatoria: 10% base).

**¿Existe overfitting, underfitting u otros problemas?**

El modelo muestra varios problemas críticos:
- **Rendimiento deficiente global**: 54.96% en entrenamiento, muy cerca de lo aleatorio (10%)
- **Gap validación-prueba pequeño**: 55.52% vs 51.67%, solo 3.85%, indicando estabilidad
- **Underfitting**: El modelo no está aprendiendo suficientemente bien ni en entrenamiento
- **Arquitectura inadecuada**: La MLP es inapropiada para datos de imagen

El problema principal no es overfitting sino que la arquitectura **no puede extraer características relevantes** de imágenes.

**¿Bucles de Retroalimentación en el Diseño

**Análisis comparativo de arquitecturas:**

| Iteración | Arquitectura | Exactitud (Train) | Exactitud (Prueba) | Observaciones |
|---|---|---|---|---|
| 1 | MLP (3072→512→256→128→10) | 54.96% | 51.67% | Rendimiento pobre, arquitectura inadecuada |
| 4 | SimpleCNN (Conv3×3 layers) | 89.76% | 82.15% | Mejora masiva: +34.8% en train, +30.5% en test |

La comparación es evidente: la CNN supera a la MLP por un margen enorme debido a que:
- La CNN respeta la estructura espacial de las imágenes
- La CNN comparte pesos reduciendo parámetros
- La CNN aprende jerarquías de características

No se implementaron mejoras iterativas a la MLP porque la limitación es fundamental a la arquitectura, no a hiperparámetros.

## Preguntas

### ¿Cuáles son las diferencias encontradas entre el modelo anterior (Ejercicio 4) y este (Ejercicio 5)?

**Análisis de cambios principales:**

| Aspecto | Ejercicio 4 (CNN) | Ejercicio 5 (MLP) |
|---|---|---|
| **Arquitectura** | 3 capas convolucionales | Red completamente conectada |
| **Respeto a estructura espacial** | Sí (kernels deslizantes) | No (aplanamiento) |
| **Número de parámetros** | ≈ 186K | ≈ 1.9M |
| **Compartición de pesos** | Sí (kernels compartidos) | No (pesos únicos) |
| **Exactitud en prueba** | **82.15%** | **51.67%** |
| **Gap Validación-Prueba** | 7.74% | 3.85% |

**Conclusiones principales:**

1. **Arquitectura apropiada es crítica**: La CNN obtiene un rendimiento ~ 30.5% mejor solo por usar la arquitectura correcta
2. **La augmentación de datos tiene efecto limitado en MLP**: Incluso con augmentación, la MLP no puede competir con CNN
3. **Eficiencia**: La CNN usa ~ 10× menos parámetros pero obtiene resultados 10× mejores
4. **Importancia de inducción de sesgo**: Las redes deben incorporar conocimiento del dominio (estructura espacial)

### ¿Generaliza bien el modelo a nuevos datos?

**Respuesta: No, el modelo no generaliza bien.**

**Análisis:**

1. **Rendimiento absoluto bajo**: Con 51.67% de exactitud, el modelo apenas supera la adivinación aleatoria (10%)
2. **Gap pequeño pero rendimiento pobre**: Aunque la diferencia entre validación (55.52%) y prueba (51.67%) es pequeña (3.85%), el rendimiento absoluto es insuficiente
3. **Extrapolación a nuevos datos**: Para imágenes nuevas que sigan CIFAR-10:
   - Rendimiento esperado: ~50-52% (consistente con prueba actual)
   - Inutilizable en práctica (necesitaría >80% para aplicaciones reales)

4. **Dominios diferentes**: Para imágenes de dominios completamente diferentes:
   - El rendimiento sería aún peor
   - La arquitectura no aprendería características transferibles

**Conclusión concluyente**: Este modelo demuestra por qué **las redes convolucionales fueron un avance revolucionario** en visión por computadora. La arquitectura MLP es fundamentalmente inapropiada para este dominio, y ninguna cantidad de tuning de hiperparámetros podría compensar esta limitación arquitectónica._plot.png)






