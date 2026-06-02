# Informe comparativo no-T4: TF-IDF, AST e hibrido

## Protocolo

Todos los modelos de esta comparacion usan los splits oficiales:

`Dataset_Java/dataset_clones_java/model_ready_balanced`

No se generan nuevos splits y no se hace balanceo adicional. La clase `T4` se
excluye de train, valid y test.

Clases activas:

`T0`, `T1`, `T2`, `VST3`, `ST3`, `MT3`, `WT3`

## Resultados en test

| Modelo | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| TF-IDF + XGBoost | 0.8185 | 0.8585 | 0.8180 |
| AST + XGBoost | 0.7346 | 0.7296 | 0.7346 |
| Hybrid TF-IDF + AST + XGBoost | 0.9134 | 0.9337 | 0.9132 |

Archivos consolidados:

`Dataset_Java/dataset_clones_java/results/model_comparison_metrics_no_t4.csv`

`Dataset_Java/dataset_clones_java/results/model_comparison_per_class_no_t4.csv`

## Interpretacion

El baseline TF-IDF mantiene un desempeno fuerte en clases con senal lexica clara
como `T0`, `T1` y `T2`, pero se debilita en clones tipo 3, especialmente `MT3` y
`WT3`.

El modelo basado solo en AST captura estructura, pero pierde informacion lexica
necesaria para separar varias clases finas.

El hibrido combina ambas senales y obtiene el mejor resultado global. Esto
respalda la decision de mantener TF-IDF + AST + XGBoost como modelo principal
del sistema.

## Evaluacion binaria de producto

El hibrido reentrenado sin `T4` tambien fue evaluado en la vista de producto:

`T0` -> `no_alert`

`T1`, `T2`, `VST3`, `ST3`, `MT3`, `WT3` -> `review_required`

| Metrica | Valor |
|---|---:|
| Accuracy | 0.9863 |
| Macro F1 | 0.9790 |
| Balanced accuracy | 0.9798 |
| Recall no_alert | 0.9689 |
| Recall review_required | 0.9908 |
| False positive rate | 0.0311 |
| F1 review_required | 0.9913 |

## Conclusion

Al excluir `T4`, la comparacion se vuelve mas adecuada para un producto
educativo de priorizacion de revision. El hibrido no solo mejora la tarea
multiclase, tambien conserva baja tasa de falsos positivos en la vista binaria.
