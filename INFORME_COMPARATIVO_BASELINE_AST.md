# Reporte Tecnico Comparativo: TF-IDF, AST e Hibrido con XGBoost

**Fase del Proyecto:** Comparacion experimental de representaciones para clasificacion multiclase de clones Java  
**Estado:** Finalizado y Validado  

---

## 1. Introduccion y Objetivo del Experimento

Este reporte compara tres pipelines de Machine Learning tradicional para la clasificacion multiclase de clones de codigo Java en las clases `T0`, `T1`, `T2`, `VST3`, `ST3`, `MT3`, `WT3` y `T4`.

El objetivo es evaluar si una representacion estructural basada en AST mejora o complementa el baseline lexico basado en TF-IDF. La hipotesis central es que las features estructurales pueden aportar senales distintas a las textuales, especialmente para clones modificados de tipo `MT3` y `WT3`, donde la similitud superficial de tokens puede ser insuficiente.

Los modelos comparados son:

| Modelo | Pipeline | Representacion | Clasificador |
| :--- | :--- | :--- | :--- |
| Baseline lexico | `baseline_xgboost_pipeline.py` | TF-IDF + features simetricas | XGBoost |
| Modelo estructural | `ast_xgboost_pipeline.py` | AST features + features simetricas | XGBoost |
| Modelo hibrido | `hybrid_xgboost_pipeline.py` | TF-IDF + AST features simetricas | XGBoost |

---

## 2. Protocolo Experimental Comun

Para que la comparacion sea justa, los tres modelos se evaluan sobre el mismo dataset balanceado y la misma particion experimental:

```text
dataset_clones_java/model_ready_balanced/
  train_balanced.jsonl
  valid_balanced.jsonl
  test_balanced.jsonl
```

Los pipelines AST e hibrido no realizan nuevo submuestreo, no generan nuevos splits y no mezclan datos de validacion o prueba durante el entrenamiento. La diferencia experimental esta controlada en la representacion de entrada al clasificador.

| Split | Cantidad de pares | Uso |
| :--- | ---: | :--- |
| Train | 12,281 | Entrenamiento del modelo |
| Valid | 2,632 | Early stopping de XGBoost |
| Test | 2,632 | Evaluacion final |
| Total | 17,545 | Dataset balanceado completo |

Los tres modelos usan `SEED = 42` y un clasificador `xgboost.XGBClassifier` con configuracion equivalente:

```python
xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric="mlogloss",
    early_stopping_rounds=15,
    tree_method="hist",
    n_jobs=-1
)
```

---

## 3. Modelo Baseline: TF-IDF + XGBoost

El baseline representa cada metodo Java como un vector TF-IDF. El vectorizador se ajusta exclusivamente con funciones del conjunto de entrenamiento para evitar fuga de informacion.

Para cada par de funciones, se construye una representacion simetrica:

```text
X_pair = [abs(v1 - v2), v1 * v2, cosine_similarity(v1, v2)]
```

Con `TFIDF_MAX_FEATURES = 1000`, el vector final tiene:

```text
1000 abs_diff + 1000 product + 1 cosine_similarity = 2001 features
```

Este modelo captura principalmente similitud lexica: tokens compartidos, diferencias de vocabulario, nombres, llamadas, tipos y patrones textuales frecuentes.

---

## 4. Modelo AST: AST Features + XGBoost

El pipeline estructural parsea cada metodo Java con `javalang`. Como el dataset contiene metodos sueltos y no clases completas, cada metodo se envuelve antes del parseo:

```java
public class Dummy {
    <method_code>
}
```

Para cada funcion se extraen 28 caracteristicas numericas:

```text
parse_success
total_nodes
max_depth
n_MethodDeclaration
n_IfStatement
n_ForStatement
n_WhileStatement
n_DoStatement
n_SwitchStatement
n_ReturnStatement
n_MethodInvocation
n_VariableDeclarator
n_LocalVariableDeclaration
n_Assignment
n_BinaryOperation
n_Literal
n_TryStatement
n_CatchClause
n_ThrowStatement
n_ClassCreator
n_MemberReference
n_StatementExpression
n_BlockStatement
n_parameters
ratio_control_flow
ratio_method_invocations
ratio_assignments
ratio_returns
```

Para cada par se construye tambien una representacion simetrica:

```text
X_pair = [abs(ast_vec1 - ast_vec2), ast_vec1 * ast_vec2, cosine_similarity, euclidean_distance]
```

Con 28 features AST por funcion, el vector final tiene:

```text
28 abs_diff + 28 product + 1 cosine_similarity + 1 euclidean_distance = 58 features
```

El pipeline es tolerante a errores: si un metodo no se puede parsear, no se detiene la ejecucion y se asigna `parse_success = 0` con el resto de features en cero.

### Tasas de parseo

| Split | Parse success rate |
| :--- | ---: |
| Train | 99.01% |
| Valid | 98.97% |
| Test | 99.13% |

Estas tasas indican que la representacion AST cubre casi todo el dataset balanceado.

---

## 5. Modelo Hibrido: TF-IDF + AST + XGBoost

El pipeline hibrido concatena las dos representaciones anteriores:

```text
X_hybrid = [TF-IDF symmetric features, AST symmetric features]
```

La dimensionalidad final queda asi:

```text
2001 TF-IDF pair features + 58 AST pair features = 2059 features
```

Esto aumenta el numero de features en solo 58 columnas frente al baseline, es decir, aproximadamente un 2.9% adicional. Como los hiperparametros de XGBoost se mantienen iguales, la mejora o degradacion observada se atribuye principalmente a la informacion estructural agregada y no a cambios de entrenamiento.

El pipeline guarda artefactos especificos en:

```text
dataset_clones_java/results/hybrid_xgboost/
  metrics.json
  classification_report.txt
  confusion_matrix.csv
  per_class_metrics.csv
  model.joblib
  label_mapping.json
```

Ademas, para facilitar graficas posteriores, se guardaron tablas comparativas en:

```text
dataset_clones_java/results/model_comparison_metrics.csv
dataset_clones_java/results/model_comparison_metrics.json
dataset_clones_java/results/model_comparison_per_class_metrics.csv
dataset_clones_java/results/model_comparison_per_class_metrics.json
```

---

## 6. Comparacion Global de Resultados

| Metrica | TF-IDF + XGBoost | AST + XGBoost | Hibrido TF-IDF + AST | Hibrido - TF-IDF |
| :--- | ---: | ---: | ---: | ---: |
| Accuracy | 84.54% | 77.74% | **92.44%** | **+7.90 pp** |
| Macro F1 | 87.30% | 76.10% | **93.82%** | **+6.52 pp** |
| Weighted F1 | 84.49% | 77.71% | **92.42%** | **+7.93 pp** |
| Feature count | 2,001 | 58 | 2,059 | +58 |

El resultado principal es claro: el hibrido supera tanto al baseline lexico como al modelo AST individual. La ganancia no viene de usar muchos mas parametros de entrada, porque solo se agregan 58 features estructurales al vector TF-IDF de 2001 dimensiones.

Esto confirma que las features AST no reemplazan a TF-IDF cuando se usan solas, pero si aportan senal complementaria cuando se combinan con la representacion lexica.

---

## 7. Comparacion por Clase

| Clase | F1 TF-IDF | F1 AST | F1 Hibrido | Hibrido - TF-IDF | Lectura |
| :--- | ---: | ---: | ---: | ---: | :--- |
| MT3 | 0.5996 | 0.7187 | **0.8086** | **+0.2090** | Mejora fuerte en clones moderados |
| ST3 | 0.8368 | 0.8834 | **0.8990** | **+0.0622** | Mejora consistente |
| T0 | **0.9680** | 0.5763 | 0.9654 | -0.0026 | Practicamente conserva el baseline |
| T1 | 0.9935 | 0.8744 | **0.9956** | +0.0021 | Ligera mejora |
| T2 | 0.9400 | 0.5658 | **0.9709** | **+0.0309** | Mejora en clones renombrados |
| T4 | 0.9967 | 0.9812 | **0.9967** | +0.0000 | Empate en desempeno maximo |
| VST3 | 0.9320 | 0.7897 | **0.9700** | **+0.0380** | Mejora clara |
| WT3 | 0.7176 | 0.6981 | **0.8996** | **+0.1820** | Mejora muy fuerte en clones debiles |

El hibrido conserva casi todo el rendimiento del baseline en clases donde TF-IDF ya era excelente (`T0`, `T1`, `T4`) y mejora precisamente las fronteras mas dificiles (`MT3`, `WT3`, `ST3`, `VST3`).

El salto mas importante esta en `MT3` y `WT3`: el modelo pasa de F1 0.5996 a 0.8086 en `MT3`, y de 0.7176 a 0.8996 en `WT3`. Esto apoya la hipotesis original: la estructura AST aporta informacion diferente a la similitud textual.

---

## 8. Analisis de Errores del Modelo Hibrido

La matriz de confusion del hibrido muestra una reduccion importante de errores frente al AST individual:

| Clase real | Aciertos hibrido | Support | Principal confusion |
| :--- | ---: | ---: | :--- |
| MT3 | 359 | 450 | `ST3` y `WT3` |
| ST3 | 414 | 450 | `MT3` |
| T0 | 432 | 450 | `WT3` y `MT3` |
| T1 | 228 | 229 | `T2` |
| T2 | 50 | 52 | `T1` y `VST3` |
| T4 | 450 | 450 | Sin errores |
| VST3 | 97 | 101 | `ST3` |
| WT3 | 403 | 450 | `MT3` y `T0` |

Los errores restantes se concentran donde la taxonomia tiene fronteras naturalmente continuas: `MT3`, `ST3` y `WT3`. Aun asi, el hibrido reduce la confusion de `T0` contra clones debiles y conserva una separacion casi perfecta en `T1`, `T2`, `T4` y `VST3`.

---

## 9. Interpretacion Experimental

Los resultados apoyan una conclusion mas fuerte que la comparacion inicial:

**TF-IDF + XGBoost es un baseline solido, pero incompleto.**  
Captura muy bien similitud lexica y separa con fuerza clases como `T0`, `T1`, `T2` y `T4`, pero sufre en clones tipo 3 moderados o debiles.

**AST + XGBoost aporta estructura, pero pierde detalle lexico cuando se usa solo.**  
El modelo AST mejora `MT3` y `ST3`, pero baja demasiado en `T0`, `T1` y `T2`. Esto muestra que la estructura por si sola es insuficiente para una clasificacion general robusta.

**El hibrido combina las fortalezas de ambos.**  
El modelo `Hybrid TF-IDF + AST + XGBoost` logra 92.44% de accuracy y 93.82% de macro F1. La mejora se obtiene agregando solo 58 features AST a las 2001 features TF-IDF, por lo que no parece un problema de exceso de parametros de entrada. Al contrario, las features estructurales agregadas tienen alta utilidad marginal.

---

## 10. Conclusiones

1. El mejor modelo individual general es el hibrido `TF-IDF + AST + XGBoost`, con 92.44% de accuracy y 93.82% de macro F1.
2. El baseline `TF-IDF + XGBoost` queda como segunda mejor opcion global, con 84.54% de accuracy y 87.30% de macro F1.
3. El modelo `AST + XGBoost` aislado no supera al baseline, pero demuestra valor complementario al mejorar `MT3` y `ST3`.
4. La mejora del hibrido frente al baseline es especialmente fuerte en `MT3` (+20.90 puntos F1) y `WT3` (+18.20 puntos F1).
5. No hay evidencia de que agregar las 58 features AST perjudique por exceso de dimensionalidad; el incremento de features es pequeno y la mejora es grande.

---

## 11. Recomendaciones para la Siguiente Iteracion

El hibrido debe mantenerse como nuevo punto de referencia tradicional del proyecto. Para estudiar si el numero de features esta optimizado, se recomienda ejecutar una pequena matriz de ablacion:

| Experimento | Cambio | Objetivo |
| :--- | :--- | :--- |
| Hibrido TF-IDF 500 | `TFIDF_MAX_FEATURES = 500` + AST | Medir si menos vocabulario conserva la mejora |
| Hibrido TF-IDF 1000 | Configuracion actual | Punto de referencia validado |
| Hibrido TF-IDF 1500/2000 | Mas features lexicas + AST | Ver si aumenta F1 en `MT3` y `WT3` |
| Hibrido sin `product` AST | Quitar producto AST | Medir aporte de interacciones estructurales |
| Hibrido con AST extendido | Agregar mas nodos/ratios | Ver si mejora sin subir mucho dimensionalidad |

Por ahora, la evidencia favorece conservar el hibrido actual: mejora sustancialmente las metricas globales y por clase con un aumento muy pequeno de features frente al baseline.

