# Reporte Tecnico Comparativo: TF-IDF + XGBoost vs AST Features + XGBoost

**Fase del Proyecto:** Comparacion experimental de representaciones para clasificacion multiclase de clones Java  
**Estado:** Finalizado y Validado  

---

## 1. Introduccion y Objetivo del Experimento

Este reporte compara dos pipelines de Machine Learning tradicional para la clasificacion multiclase de clones de codigo Java en las clases `T0`, `T1`, `T2`, `VST3`, `ST3`, `MT3`, `WT3` y `T4`.

El objetivo del experimento es evaluar si una representacion estructural basada en caracteristicas de AST puede mejorar o complementar el baseline lexico basado en TF-IDF. La hipotesis principal es que las caracteristicas estructurales pueden aportar senales distintas a las textuales, especialmente en clones modificados de tipo `MT3` y `WT3`, donde la similitud superficial de tokens puede ser insuficiente.

Los modelos comparados son:

| Modelo | Pipeline | Representacion | Clasificador |
| :--- | :--- | :--- | :--- |
| Baseline lexico | `baseline_xgboost_pipeline.py` | TF-IDF + features simetricas | XGBoost |
| Modelo estructural | `ast_xgboost_pipeline.py` | AST features + features simetricas | XGBoost |

---

## 2. Protocolo Experimental Comun

Para que la comparacion sea justa, ambos modelos se evaluan sobre el mismo dataset balanceado y la misma particion experimental:

```text
dataset_clones_java/model_ready_balanced/
  train_balanced.jsonl
  valid_balanced.jsonl
  test_balanced.jsonl
```

El pipeline AST no realiza nuevo submuestreo, no genera nuevos splits y no mezcla datos de validacion o prueba durante el entrenamiento. La unica diferencia experimental buscada es la representacion de entrada al clasificador.

| Split | Cantidad de pares | Uso |
| :--- | ---: | :--- |
| Train | 12,281 | Entrenamiento del modelo |
| Valid | 2,632 | Early stopping de XGBoost |
| Test | 2,632 | Evaluacion final |
| Total | 17,545 | Dataset balanceado completo |

Ambos modelos usan `SEED = 42` y un clasificador `xgboost.XGBClassifier` con configuracion equivalente:

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

El nuevo pipeline estructural parsea cada metodo Java con `javalang`. Como el dataset contiene metodos sueltos y no clases completas, cada metodo se envuelve antes del parseo:

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

El pipeline esta disenado para ser tolerante a errores: si un metodo no se puede parsear, no se detiene la ejecucion y se asigna `parse_success = 0` con el resto de features en cero.

### Tasas de parseo

| Split | Parse success rate |
| :--- | ---: |
| Train | 99.01% |
| Valid | 98.97% |
| Test | 99.13% |

Estas tasas indican que la representacion AST cubre casi todo el dataset balanceado. Los fallos restantes probablemente corresponden a metodos incompletos, fragmentos con dependencias sintacticas externas o casos que `javalang` no puede interpretar como metodo Java valido aun despues del wrapping.

---

## 5. Comparacion Global de Resultados

| Metrica | TF-IDF + XGBoost | AST + XGBoost | Diferencia AST - TF-IDF |
| :--- | ---: | ---: | ---: |
| Accuracy | 84.54% | 77.74% | -6.80 pp |
| Macro F1 | 87.30% | 76.10% | -11.20 pp |
| Weighted F1 | 84.49% | 77.71% | -6.78 pp |
| Feature count | 2,001 | 58 | -1,943 |

El baseline lexico obtiene mejor desempeno global. Esto sugiere que, para este dataset balanceado, la informacion textual capturada por TF-IDF sigue siendo una senal muy fuerte para distinguir la mayoria de clases.

Sin embargo, el modelo AST no queda descartado: con solo 58 features estructurales logra un desempeno competitivo y mejora clases clave de tipo 3. Esto apunta a que el AST no necesariamente reemplaza al baseline lexico, sino que puede complementarlo.

---

## 6. Comparacion por Clase

| Clase | F1 TF-IDF | F1 AST | Diferencia AST - TF-IDF | Lectura |
| :--- | ---: | ---: | ---: | :--- |
| MT3 | 0.5996 | 0.7187 | +0.1191 | Mejora clara con estructura |
| ST3 | 0.8368 | 0.8834 | +0.0466 | Mejora moderada con estructura |
| T0 | 0.9680 | 0.5763 | -0.3917 | TF-IDF separa mucho mejor no-clones |
| T1 | 0.9935 | 0.8744 | -0.1191 | TF-IDF captura mejor clones exactos |
| T2 | 0.9400 | 0.5658 | -0.3742 | TF-IDF captura mejor renombramientos |
| T4 | 0.9967 | 0.9812 | -0.0155 | Ambos modelos rinden muy alto |
| VST3 | 0.9320 | 0.7897 | -0.1423 | TF-IDF mantiene ventaja |
| WT3 | 0.7176 | 0.6981 | -0.0195 | Rendimiento similar, ligera ventaja TF-IDF |

El hallazgo mas relevante es que AST mejora `MT3` y `ST3`, precisamente dos clases donde la estructura del codigo puede ser mas informativa que el solapamiento textual puro. En `MT3`, el F1 sube de 0.5996 a 0.7187, una mejora de 11.91 puntos.

La perdida mas fuerte del AST ocurre en `T0`, `T2` y `T1`. Esto es esperable: los no-clones suelen diferenciarse por vocabulario y contexto lexico, mientras que clones exactos o renombrados conservan patrones textuales que TF-IDF detecta con mucha facilidad. Las features AST agregadas, al ser conteos estructurales globales, pueden considerar similares dos metodos que tienen forma parecida aunque implementen logicas distintas.

---

## 7. Analisis de Errores del Modelo AST

La matriz de confusion del modelo AST muestra los siguientes patrones principales:

| Clase real | Error principal | Interpretacion |
| :--- | :--- | :--- |
| T0 | Confusion con `WT3` y `MT3` | Metodos no relacionados pueden compartir forma estructural general |
| WT3 | Confusion con `T0` y `MT3` | La frontera entre no-clon, clon debil y clon moderado sigue siendo difusa |
| T1 | Confusion con `T2` | La estructura es casi identica entre clones exactos y renombrados |
| VST3 | Confusion con `T2` | Conteos AST no capturan cambios finos de expresiones o nombres |
| MT3 | Mejora frente al baseline | La forma estructural ayuda cuando hay modificaciones sintacticas moderadas |

En concreto, el modelo AST predice correctamente 336 de 450 ejemplos `MT3` y 409 de 450 ejemplos `ST3`, lo cual explica la mejora frente al baseline en estas clases. En cambio, solo clasifica correctamente 238 de 450 ejemplos `T0`, porque muchos no-clones comparten una silueta estructural parecida con clones debiles o moderados.

---

## 8. Interpretacion Experimental

Los resultados apoyan una conclusion matizada:

**TF-IDF + XGBoost es mejor como modelo individual general.**  
El baseline obtiene mejor accuracy, macro F1 y weighted F1. La informacion lexica es especialmente fuerte para `T0`, `T1`, `T2`, `T4` y `VST3`.

**AST + XGBoost aporta informacion complementaria.**  
El modelo AST mejora `MT3` y `ST3`, dos clases relevantes para clones modificados tipo 3. Esto confirma que las senales estructurales capturan patrones que TF-IDF no explota por completo.

**AST features actuales son compactas pero limitadas.**  
El vector AST usa solo 58 features finales frente a 2001 del baseline. Esta compacidad facilita interpretabilidad y bajo costo, pero tambien pierde detalles finos: nombres de metodos, operadores concretos, literales especificos, orden de sentencias y relaciones de flujo de datos.

---

## 9. Conclusiones

1. El baseline `TF-IDF + XGBoost` sigue siendo el mejor modelo individual en desempeno global, con 84.54% de accuracy y 87.30% de macro F1.
2. El modelo `AST + XGBoost` alcanza 77.74% de accuracy y 76.10% de macro F1 usando solo 58 features, lo cual confirma que la estructura del codigo contiene senal predictiva relevante.
3. La mejora del AST en `MT3` (+11.91 puntos F1) y `ST3` (+4.66 puntos F1) es el resultado mas importante del experimento.
4. La representacion AST actual no reemplaza al baseline lexico, pero si parece util como complemento.
5. El siguiente paso natural es entrenar un modelo hibrido que concatene `TF-IDF pair features` y `AST pair features`, manteniendo exactamente los mismos splits balanceados.

---

## 10. Recomendacion para la Siguiente Iteracion

Se recomienda implementar un tercer pipeline experimental:

```text
hybrid_xgboost_pipeline.py
```

con la representacion:

```text
X_hybrid = [TF-IDF symmetric features, AST symmetric features]
```

Esto permitiria probar directamente si las ganancias del AST en `MT3` y `ST3` pueden conservarse sin sacrificar el excelente rendimiento del baseline en `T0`, `T1`, `T2`, `T4` y `VST3`.

La expectativa experimental es que el modelo hibrido supere al baseline lexico en clases tipo 3 y mantenga una exactitud global cercana o superior a TF-IDF + XGBoost.

