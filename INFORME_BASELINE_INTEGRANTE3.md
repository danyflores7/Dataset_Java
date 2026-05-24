# Reporte Técnico: Pipeline Baseline de Clasificación Multiclase de Clones de Código en Java

**Rol:** Integrante 3 — Senior MLOps Engineer & ML Researcher  
**Fase del Proyecto:** Implementación de Baseline & Validación de Modelo de Clasificación  
**Estado:** Finalizado y Validado  

---

## 1. Introducción y Contexto de la Tarea (Integrante 3)

La detección automática de clones de código (fragmentos duplicados o funcionalmente equivalentes) es fundamental en la ingeniería de software moderna para mitigar la deuda técnica y prevenir la propagación de fallos. Este reporte formaliza el diseño, implementación y evaluación del **pipeline baseline de clasificación multiclase** para clasificar pares de métodos en Java según la taxonomía estándar de clones (`T0`, `T1`, `T2`, `VST3`, `ST3`, `MT3`, `WT3`, `T4`).

El objetivo primordial de construir este modelo baseline usando algoritmos de Machine Learning tradicional (específicamente **XGBoost**) es establecer una línea base cuantitativa y de bajo costo computacional. Este punto de partida servirá para contrastar y medir el valor agregado real de modelos de lenguaje pre-entrenados más complejos (tales como *Transformers* de solo codificación: CodeBERT, GraphCodeBERT o UniXcoder) en etapas posteriores.

### El Desafío del Desbalance de Clases
El dataset unificado (que integra a gran escala *BigCloneBench* y *GPTCloneBench*) consta de **1,739,359 pares**. No obstante, presenta una distribución de clases realista pero sumamente desbalanceada y extrema. Las clases no-clones (`T0`: 1.17M) y clones débiles (`WT3`: 379k) dominan abrumadoramente el corpus, mientras que las categorías minoritarias críticas (`T1`, `T2`, `VST3`) representan menos del 0.2% del volumen total. Sin una estrategia rigurosa de control de desbalance, cualquier clasificador colapsaría prediciendo la clase mayoritaria (Paradoja de la Exactitud), invalidando su uso práctico.

---

## 2. La Estrategia de Ingeniería de Datos (Control de Desbalance)

Para entrenar el modelo baseline de forma viable en entornos locales o de desarrollo sin incurrir en desbordamientos de memoria (**Out-of-Memory (OOM)**), fue necesario diseñar una estrategia de ingesta altamente eficiente. El dataset unificado completo en formato JSON Lines (`.jsonl`) pesa aproximadamente **5.7 GB** en disco. Cargar la totalidad del texto de los fragmentos de código Java directamente a memoria RAM en Python colapsaría el entorno de ejecución habitual.

Para solucionar este inconveniente físico, implementamos un algoritmo de **submuestreo en dos pasadas consecutivas (Two-Pass Index-Streaming)**:

1.  **Primera Pasada (Escaneo de Estructura):** Se realiza un *stream* línea por línea de los archivos JSONL. En lugar de decodificar todo el objeto JSON (que contiene extensas cadenas de texto de código Java), se utiliza una función de búsqueda optimizada a nivel de bytes (`line.rfind`) para extraer únicamente la etiqueta de la clase (`clone_type`) y registrar su ubicación física (archivo origen e índice de la línea). Este proceso es sumamente veloz y consume apenas $\approx 50$ MB de memoria RAM.
2.  **Fase de Submuestreo (Undersampling):** Utilizando los índices mapeados, se define un límite estricto de **$MAX\_SAMPLES\_PER\_CLASS = 3000$**. Aquellas clases cuya presencia supere este umbral (tales como `T0`, `WT3`, `MT3` y `T4`) son submuestreadas aleatoriamente mediante `random.sample`. Las clases minoritarias con cantidades inferiores al umbral se conservan en su totalidad para maximizar el aprendizaje del clasificador en estas categorías.
3.  **Segunda Pasada (Carga Selectiva):** Los índices seleccionados se ordenan de manera ascendente por cada archivo. Posteriormente, se realiza una sola lectura lineal secuencial de los archivos `.jsonl`, decodificando el objeto JSON completo (con el texto de `func1` y `func2`) únicamente para las líneas seleccionadas. Esto minimiza el uso de CPU y garantiza que en memoria solo residan los fragmentos de código del subconjunto balanceado definitivo.
4.  **Exportación a Disco y Compartición (GitHub):** Tras generar la partición estratificada de este subconjunto balanceado, el pipeline guarda los archivos resultantes en la carpeta `dataset_clones_java/model_ready_balanced/` (`train_balanced.jsonl`, `valid_balanced.jsonl`, `test_balanced.jsonl`). Con un peso total sumado de apenas **~43 MB**, este dataset balanceado final es ideal para ser subido a GitHub sin restricciones de tamaño y compartido directamente con el resto del equipo de desarrollo.

---

## 3. Tabla de Distribución de Datos (Antes vs. Después del Balanceo)

A continuación, se detalla la comparación cuantitativa de la cantidad de muestras por categoría en el corpus original y el corpus balanceado utilizado para entrenar y evaluar el baseline:

| Etiqueta (clone_type) | Descripción | Cantidad Original | Cantidad Balanceada (Baseline) | Cambio / Estado |
| :--- | :--- | :--- | :--- | :--- |
| **T0** | Negativos / No-Clones | 1,170,339 | 3,000 | Submuestreado (Cap de 3K) |
| **T1** | Exact Clones | 1,531 | 1,531 | Conservado Completo |
| **T2** | Renamed Clones | 342 | 342 | Conservado Completo |
| **VST3** | Very Strong Type-3 ($\geq 90\%$) | 672 | 672 | Conservado Completo |
| **ST3** | Strong Type-3 ($70\% - 90\%$) | 9,711 | 3,000 | Submuestreado (Cap de 3K) |
| **MT3** | Moderate Type-3 ($50\% - 70\%$) | 170,053 | 3,000 | Submuestreado (Cap de 3K) |
| **WT3** | Weak Type-3 ($< 50\%$) | 379,212 | 3,000 | Submuestreado (Cap de 3K) |
| **T4** | Semantic Clones | 7,499 | 3,000 | Submuestreado (Cap de 3K) |
| **TOTAL** | **Todos los pares** | **1,739,359** | **17,545** | **Reducción del 98.99% en tamaño** |

---

## 4. Extracción de Características y Vectorización

Una vez obtenido el dataset balanceado de 17,545 pares, procedimos a transformar la representación textual de las funciones Java en vectores tabulares numéricos de la siguiente manera:

1.  **Ajuste del Vectorizador (TF-IDF):** Ajustamos un modelo `TfidfVectorizer` utilizando únicamente el conjunto de funciones combinadas del split de entrenamiento (evitando fugas de datos o *data leakage*). Se configuró un límite de **1,000 características** (`max_features=1000`) y un patrón de tokenización estándar a nivel de palabras para extraer el vocabulario característico del lenguaje Java.
2.  **Vectores Individuales:** Para cada par, se transforman de forma independiente los fragmentos de código $f_1$ y $f_2$ en sus correspondientes vectores densos de pesos TF-IDF, denotados como $v_1$ y $v_2 \in \mathbb{R}^{1000}$.
3.  **Generación de Características Simétricas:** Para que la clasificación de un par de códigos sea **invariante al orden de entrada** (es decir, que la predicción para $(f_1, f_2)$ sea matemáticamente idéntica a la de $(f_2, f_1)$), el vector de características resultante $X_{combined}$ se construye empleando operadores simétricos:
    *   **Diferencia Absoluta:**
        $$d_{abs} = |v_1 - v_2| \quad (\text{dimensión: } 1000)$$
        Mide la disparidad textual absoluta en el vocabulario de ambos códigos.
    *   **Multiplicación Elemento a Elemento (Producto de Hadamard):**
        $$p_{elem} = v_1 \odot v_2 \quad (\text{dimensión: } 1000)$$
        Destaca los términos y tokens que se comparten simultáneamente en ambos códigos.
    *   **Similitud del Coseno:**
        $$\text{cosine\_sim} = \frac{v_1 \cdot v_2}{\|v_1\| \|v_2\| + \epsilon} \quad (\text{dimensión: } 1)$$
        Representa la cercanía angular global de las estructuras de código en el espacio vectorial.

El vector de entrada definitivo para alimentar a XGBoost resulta de la concatenación horizontal de estas componentes:
$$X_{combined} = [d_{abs} \,\|\, p_{elem} \,\|\, \text{cosine\_sim}] \quad \in \mathbb{R}^{2001}$$

---

## 5. Entrenamiento y Partición (Splits)

Para garantizar evaluaciones confiables y reproducibles, estructuramos la partición del dataset con las siguientes directrices:

*   **Partición Estratificada Global (70% / 15% / 15%):** El conjunto de datos balanceado se dividió utilizando una estratificación basada en el tipo de clon (`clone_type`). Esto garantiza que la proporción de clases se mantenga constante a través de las particiones:
    *   **Entrenamiento (Train):** 12,281 muestras (usado para ajustar el TF-IDF y el modelo).
    *   **Validación (Valid):** 2,632 muestras (usado para control de *early stopping* en XGBoost).
    *   **Prueba (Test):** 2,632 muestras (usado para evaluación ciega final).
*   **Reproducibilidad Absoluta:** Fijamos una semilla aleatoria global (**`SEED = 42`**) tanto para la división de los splits (`train_test_split`) como para la inicialización del estimador de árboles de decisión en XGBoost. Esto asegura que el experimento genere métricas exactamente idénticas bajo ejecuciones sucesivas.

---

## 6. Métricas de Evaluación del Baseline (XGBClassifier)

El clasificador multiclase `xgboost.XGBClassifier` se entrenó empleando el método de histograma rápido (`tree_method='hist'`) con una tasa de aprendizaje de $0.1$ y parada temprana (*early stopping*) de 15 iteraciones sobre la pérdida de entropía cruzada multiclase (`mlogloss`) de validación. La iteración óptima se obtuvo en la ronda 271.

A continuación, se presentan los resultados arrojados por el modelo en el conjunto de prueba (Test Split):

### Métricas Globales
*   **Exactitud (Accuracy) en Test:** **84.54%**
*   **Macro F1-Average:** **87.30%**
*   **Weighted F1-Average:** **84.49%**

### Desglose por Clase (Clone Type)
| Clase (Target) | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **MT3** | 0.5905 | 0.6089 | 0.5996 | 450 |
| **ST3** | 0.8251 | 0.8489 | 0.8368 | 450 |
| **T0** | 0.9626 | 0.9733 | 0.9680 | 450 |
| **T1** | 0.9871 | 1.0000 | 0.9935 | 229 |
| **T2** | 0.9792 | 0.9038 | 0.9400 | 52 |
| **T4** | 0.9956 | 0.9978 | 0.9967 | 450 |
| **VST3** | 0.9143 | 0.9505 | 0.9320 | 101 |
| **WT3** | 0.7488 | 0.6889 | 0.7176 | 450 |

### Análisis de Rendimiento
1.  **Detección Óptima de Clones Claros (`T1`, `T2`, `T4`, `T0`):**
    *   **T1 (Exactos) y T4 (Semánticos):** Obtienen un F1-score excepcional de **99.35%** y **99.67%** respectivamente. Esto se debe a que los clones de tipo 1 comparten casi idéntica huella de tokens TF-IDF, y las muestras semánticas de GPTCloneBench exhiben patrones y firmas estructurales muy compactas y diferenciadas en comparación con los clones sintácticos tradicionales de BigCloneBench.
    *   **T0 (No-clones):** Muestra una excelente tasa de separación con un F1-score de **96.80%**, logrando discriminar eficazmente pares que no guardan relación funcional.
2.  **El Cuello de Botella Sintáctico (`MT3`):**
    *   El modelo clasifica con dificultad los clones sintácticos de nivel moderado (**`MT3`**), alcanzando un F1-score de apenas **59.96%**. Esto se debe a que `MT3` se encuentra en la frontera difusa de similitud de tokens ($50\%$ a $70\%$). Al utilizar vectorización puramente léxica (TF-IDF), el modelo carece de comprensión semántica contextual profunda o conocimiento estructural (como gráficos de flujo de datos) necesarios para dirimir si una baja similitud de tokens corresponde a un clon moderado (`MT3`), a un clon débil (`WT3`) o a un no-clon (`T0`).
3.  **Área de Mejora Futura:**
    *   La debilidad en la frontera sintáctica de tipo 3 (`MT3`, `WT3`) fundamenta sólidamente la necesidad de migrar hacia arquitecturas basadas en Deep Learning y Transformers de Código (por ejemplo, embeddings contextuales de **CodeBERT**), las cuales capturan representaciones semánticas más ricas e independientes del conteo de términos léxicos.
