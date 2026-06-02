# Informe baseline TF-IDF + XGBoost no-T4

## Objetivo

Este reporte documenta el baseline tradicional TF-IDF + XGBoost para
clasificacion multiclase de pares de metodos Java.

El baseline usa los splits oficiales ya existentes y excluye `T4` de todo
entrenamiento y evaluacion. No genera nuevos splits ni aplica balanceo
adicional.

## Clases activas

`T0`, `T1`, `T2`, `VST3`, `ST3`, `MT3`, `WT3`

`T4` queda excluido del protocolo actual.

## Configuracion

| Elemento | Valor |
|---|---|
| Fuente | `model_ready_balanced/train_balanced.jsonl`, `valid_balanced.jsonl`, `test_balanced.jsonl` |
| Vectorizador | TF-IDF |
| Max features | 1000 |
| Features por par | `abs_diff`, `product`, `cosine_similarity` |
| Clasificador | XGBoost multiclass |
| Seed | 42 |

## Tamano de splits

| Split | Original | Excluidos T4 | Activo |
|---|---:|---:|---:|
| Train | 12,281 | 2,100 | 10,181 |
| Valid | 2,632 | 450 | 2,182 |
| Test | 2,632 | 450 | 2,182 |

## Resultados en test

| Clase | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| MT3 | 0.6044 | 0.6111 | 0.6077 | 450 |
| ST3 | 0.8067 | 0.8533 | 0.8294 | 450 |
| T0 | 0.9733 | 0.9733 | 0.9733 | 450 |
| T1 | 0.9913 | 1.0000 | 0.9957 | 229 |
| T2 | 0.9792 | 0.9038 | 0.9400 | 52 |
| VST3 | 0.9143 | 0.9505 | 0.9320 | 101 |
| WT3 | 0.7602 | 0.7044 | 0.7313 | 450 |

| Metrica | Valor |
|---|---:|
| Accuracy | 0.8185 |
| Macro F1 | 0.8585 |
| Weighted F1 | 0.8180 |

## Interpretacion

TF-IDF + XGBoost funciona bien como baseline economico y fuerte para clases con
senal lexica clara. Sus principales debilidades aparecen en clones modificados
tipo 3, sobre todo `MT3` y `WT3`, donde la similitud superficial deja de ser
suficiente.

Este comportamiento justifica comparar contra representaciones mas ricas como
AST e embeddings de CodeBERT, y tambien explica por que el hibrido TF-IDF + AST
mejora el resultado global.

## Artefactos

`Dataset_Java/dataset_clones_java/results/tfidf_xgboost/metrics.json`

`Dataset_Java/dataset_clones_java/results/tfidf_xgboost/classification_report.txt`

`Dataset_Java/dataset_clones_java/results/tfidf_xgboost/confusion_matrix.csv`

`Dataset_Java/dataset_clones_java/results/tfidf_xgboost/per_class_metrics.csv`
