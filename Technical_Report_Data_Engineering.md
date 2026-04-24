# Reporte Técnico: Ingeniería de Datos para Detección de Clones en Código Java

**Rol:** Lead Data QA Engineer / ML Researcher  
**Fase del Proyecto:** Data Engineering & Pipeline de Estandarización  

---

## 1. Introducción y Justificación del Problema

La detección de código duplicado, o "clones de código", es una tarea crítica en el mantenimiento y la evolución del software, orientada a reducir la deuda técnica y prevenir la propagación de errores. Tradicionalmente, las herramientas léxicas y basadas en el Árbol de Sintaxis Abstracta (AST) han demostrado gran eficacia detectando copias textuales exactas o parametrizadas. Sin embargo, estas metodologías sufren limitaciones severas al enfrentar **clones semánticos** (fragmentos de código que realizan la misma funcionalidad pero con estructuras algorítmicas, sintaxis o APIs completamente diferentes).

Para abordar la detección de clones semánticos, el Aprendizaje Automático (ML), y en particular los modelos de lenguaje pre-entrenados sobre código (como CodeBERT o GraphCodeBERT), representa la solución del estado del arte. Estos modelos son capaces de capturar el contexto profundo y la semántica subyacente. No obstante, para entrenar modelos robustos que generalicen más allá de la coincidencia estructural, es imperativo contar con un corpus de entrenamiento a gran escala, etiquetado meticulosamente con alta granularidad y que exponga al modelo a la máxima variabilidad algorítmica posible.

## 2. Selección de Fuentes de Datos

Para lograr un corpus representativo de la realidad del desarrollo en Java, se diseñó un enfoque de unificación de conjuntos de datos:

*   **BigCloneBench (vía CodeXGLUE):** Proveedor principal de clones extraídos del repositorio inter-proyectos *IJaDataset*. Representa la base del estado del arte para evaluación de clones debido a su minado exhaustivo y verificación humana. Dado que la distribución de CodeXGLUE proporciona únicamente pares con etiquetas binarias (Clon / No Clon), fue **imprescindible** realizar una extracción directa desde la base de datos relacional nativa (`bcb.h2.db`) usando un driver JDBC en Java. Este paso de recuperación de metadatos nos permitió obtener los identificadores exactos y la similitud a nivel de tokens, devolviéndonos la dimensionalidad de las etiquetas originales perdidas en la compresión de CodeXGLUE.
*   **GPTCloneBench:** Dataset moderno enfocado exclusivamente en **clones semánticos (Tipo 4)** inyectados o validados mediante Modelos de Lenguaje Grandes (LLMs). La integración de este conjunto incrementa críticamente el rendimiento del modelo en las capas de abstracción funcional.

## 3. Taxonomía de Clones

El corpus unificado se adhiere a la siguiente taxonomía estándar en la literatura académica:

*   **T1 (Type-1 / Exact Clones):** Fragmentos de código idénticos, admitiendo únicamente variaciones en espacios en blanco, tabulaciones y comentarios.
*   **T2 (Type-2 / Renamed Clones):** Fragmentos estructural y sintácticamente idénticos, donde varían los nombres de identificadores, variables, tipos, y constantes.
*   **T3 (Type-3 / Gapped Clones):** Código con modificaciones a nivel de sentencias (inserción, borrado, modificación de líneas). Su clasificación se subdivide basándose en la métrica de Similitud de Tokens (`SIMILARITY_TOKEN`):
    *   **VST3 (Very Strongly Type-3):** Similitud $\geq 90\%$.
    *   **ST3 (Strongly Type-3):** Similitud $\geq 70\%$ y $< 90\%$.
    *   **MT3 (Moderately Type-3):** Similitud $\geq 50\%$ y $< 70\%$.
    *   **WT3 (Weakly Type-3):** Similitud $< 50\%$. Se comportan en la frontera de los clones semánticos.
*   **T4 (Type-4 / Semantic Clones):** Fragmentos con sintaxis o algoritmos totalmente distintos, pero que ejecutan el mismo comportamiento funcional u objetivo semántico. Aportados íntegramente por la integración de GPTCloneBench y la franja inferior de BigCloneBench.
*   **T0 (Negativos / No Clones):** Pares de fragmentos que no comparten ni estructura ni comportamiento semántico. Constituyen la clase negativa necesaria para el aprendizaje discriminativo.

## 4. Arquitectura del Pipeline de Datos (El Proceso)

El pipeline de ingeniería de datos se diseñó para transformar fuentes dispares crudas en un estándar consumible por arquitecturas de redes neuronales:

1.  **Ingesta y Extracción de Metadatos H2:** Se programó un extractor Java (`ExtractH2.java`) que interactúa con la base relacional original de BigCloneBench para generar el archivo maestro `clones.csv`, exponiendo los verdaderos tipos (T1-T3) e índices de similitud para millones de pares.
2.  **Operación de *Inner Join* y Unificación:** Mediante un script en Python (`build_unified_dataset.py`), se realizó un *join* en memoria cruzando los identificadores de texto de CodeXGLUE con los registros extraídos del CSV, mapeando así el texto del código a su taxonomía granular.
3.  **Procesamiento de LLM Mocks:** Se iteraron los directorios de GPTCloneBench, realizando un particionado de cadenas (`split_java_functions`) para transformar archivos únicos de texto en pares distinguibles (`func1` y `func2`), etiquetándolos estáticamente como `T4`.
4.  **Generación de Corpus Bi-Encoder (JSONL):** Se aplicó una permutación (*shuffle*) pseudoaleatoria global y el conjunto total se particionó en formato JSON Lines (`train.jsonl`, `valid.jsonl`, `test.jsonl`) manteniendo un esquema homogéneo: `{"id1", "id2", "func1", "func2", "label", "clone_type"}`.

## 5. Análisis de la Distribución de Datos (Class Imbalance)

Tras la unificación, la auditoría de calidad de datos generó el siguiente reporte global en los conjuntos combinados:

| Etiqueta (clone_type) | Cantidad de Pares |
|-----------------------|-------------------|
| MT3                   | 170,053           |
| ST3                   | 9,711             |
| T0                    | 1,170,339         |
| T1                    | 1,531             |
| T2                    | 342               |
| T4                    | 7,499             |
| VST3                  | 672               |
| WT3                   | 379,212           |
| **TOTAL GLOBAL**      | **1,739,359**     |

### 5.1 Justificación del Desbalance Natural y Mitigación Estratégica

La matriz expone un **fuerte desbalance de clases**, fuertemente dominado por pares Negativos (`T0`: 1.17M) y Weakly Type-3 (`WT3`: 379k), con una representación minoritaria en clones tempranos (`T1`, `T2`, `VST3`).

**Justificación Técnica:** Este fenómeno no es una anomalía del minado, sino una **representación fiel de la entropía en repositorios de software del mundo real**. La probabilidad estocástica de que dos funciones aleatorias realicen tareas idénticas es infinitesimal en comparación con la de pares inconexos (`T0`). Asimismo, en la evolución de proyectos consolidados, los desarrolladores corrigen o refactorizan copias exactas (`T1`, `T2`) mediante abstracciones, pero el código tiende a mutar generando densos volúmenes de código moderadamente similar que hace tareas diferentes, o código distinto que hace tareas similares (`WT3` / `T4`).

**Recomendaciones para la Fase de Modelado:**
Para prevenir que la red neuronal colapse en la predicción mayoritaria (Accuracy Paradox), se requiere la intervención del Científico de Datos durante el bucle de entrenamiento. Se recomiendan dos estrategias complementarias:
1.  **Undersampling Dinámico (Mini-Batching):** Construir un *DataLoader* que garantice que cada batch de entrenamiento (ej. tamaño 32) contenga una distribución forzada e idéntica de clones positivos y negativos, submuestreando de forma aleatoria los 1.1M de T0 por cada época.
2.  **Weighted Loss Functions (Class Weights):** Si se aproxima el problema como clasificación multicategórica o contraste, se deben ponderar los gradientes inversamente proporcionales a las frecuencias de clase (penalizando más el fallo al detectar un esquivo T4 o T2 que un abundante T0) utilizando entropía cruzada ponderada (*Weighted Cross-Entropy Loss*) o la pérdida focal (*Focal Loss*).

## 6. Conclusión

El pipeline orquestado ha culminado en un robusto activo de datos de 1.7 millones de muestras curadas. El uso del formato estandarizado **JSONL (JSON Lines)** con un esquema par-etiqueta posiciona a este dataset de forma inmejorable para arquitecturas transformadoras **bi-encoder** y **cross-encoder**. Modelos fundacionales como **CodeBERT** o **UniXcoder** pueden consumir este formato iterativamente para optimizar incrustaciones de vectores (embeddings) densos que discriminen eficazmente no solo la topología del código, sino el verdadero propósito del programador expresado a través del lenguaje Java.
