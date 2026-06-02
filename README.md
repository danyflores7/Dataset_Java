# Dataset Java - Code Clone Detection

Repositorio enfocado en la fase de **Ingeniería de Datos** para la construcción de un dataset unificado de detección de clones en código Java. Este trabajo combina de forma estructurada los repositorios *BigCloneBench (CodeXGLUE)* y *GPTCloneBench* en un formato `JSONL` listo para el entrenamiento de modelos Bi-Encoder (ej. CodeBERT).

> Nota de protocolo: los modelos activos y reportes comparativos actuales excluyen `T4` de train, valid y test. `T4` permanece documentado como parte de la taxonomia/dataset, pero no se usa en la comparacion principal de modelos.

## 📂 Enlaces a los Datos Pesados

Debido a su gran volumen, el dataset final (`model_ready`) y los binarios de la base de datos no están alojados en este repositorio de GitHub. 
Puedes descargarlos desde nuestro Google Drive:

* 📥 **[Descargar Dataset Unificado y Metadatos desde Google Drive](https://drive.google.com/drive/folders/1KDpjiXUlyrYBGuBHJXiC0igVgfivfgJZ?usp=sharing)**

## 📚 Documentación

Revisa los siguientes reportes para entender la arquitectura y la distribución de los datos:
1. **[Reporte Técnico de Data Engineering](Technical_Report_Data_Engineering.md):** Arquitectura del pipeline, taxonomía de clones (T1, T2, T3, T4) y justificación del desbalance de clases.
2. **[Reporte de Validación QA](QA_Validation_Report.md):** Conteo exacto de métricas y muestreo aleatorio con extractos de código reales.

## ⚙️ Contenido del Repositorio

* `dataset_clones_java/metadata/ExtractH2.java`: Script Java que extrae las etiquetas reales desde la base de datos relacional H2 de BigCloneBench.
* `dataset_clones_java/build_unified_dataset.py`: Pipeline principal en Python que unifica todas las fuentes, genera el *inner join* con los metadatos y crea las particiones de Train/Valid/Test.
* `dataset_clones_java/qa_validator.py`: Script de auditoría que evalúa la distribución de clases en el dataset final y genera reportes en Markdown.
