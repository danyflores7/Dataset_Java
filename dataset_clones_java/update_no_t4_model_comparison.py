#!/usr/bin/env python3
"""
Build a no-T4 comparison table across available model results.

CodeBERT is included through its post-hoc no-T4 evaluation until full embeddings
are regenerated and the model can be retrained without T4.
"""

import csv
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

MODEL_SPECS = [
    {
        "model": "TF-IDF + XGBoost",
        "metrics": RESULTS_DIR / "tfidf_xgboost" / "metrics.json",
        "per_class": RESULTS_DIR / "tfidf_xgboost" / "per_class_metrics.csv",
    },
    {
        "model": "AST + XGBoost",
        "metrics": RESULTS_DIR / "ast_xgboost" / "metrics.json",
        "per_class": RESULTS_DIR / "ast_xgboost" / "per_class_metrics.csv",
    },
    {
        "model": "Hybrid TF-IDF + AST + XGBoost",
        "metrics": RESULTS_DIR / "hybrid_xgboost" / "metrics.json",
        "per_class": RESULTS_DIR / "hybrid_xgboost" / "per_class_metrics.csv",
    },
    {
        "model": "CodeBERT + XGBoost (post-hoc no T4)",
        "metrics": RESULTS_DIR / "codebert_xgboost" / "metrics_no_t4_eval.json",
        "per_class": RESULTS_DIR / "codebert_xgboost" / "per_class_metrics_no_t4_eval.csv",
    },
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def normalize_per_class_row(model_name, row):
    return {
        "model": model_name,
        "class": row.get("class", ""),
        "precision": row.get("precision", ""),
        "recall": row.get("recall", ""),
        "f1_score": row.get("f1_score", row.get("f1-score", "")),
        "support": row.get("support", ""),
    }


def main():
    metric_rows = []
    per_class_rows = []

    for spec in MODEL_SPECS:
        if not spec["metrics"].exists():
            continue

        metrics = load_json(spec["metrics"])
        metric_rows.append(
            {
                "model": spec["model"],
                "evaluation": metrics.get("evaluation", "multiclass_no_t4"),
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "weighted_f1": metrics.get("weighted_f1"),
                "train_size": metrics.get("train_size", ""),
                "valid_size": metrics.get("valid_size", ""),
                "test_size": metrics.get("test_size", ""),
                "excluded_classes": ",".join(metrics.get("excluded_classes", metrics.get("excluded_actual_classes", []))),
                "posthoc_filter": metrics.get("posthoc_filter", False),
                "retrained_without_t4": not metrics.get("posthoc_filter", False),
                "metrics_file": str(spec["metrics"].relative_to(RESULTS_DIR)),
            }
        )

        if spec["per_class"].exists():
            with open(spec["per_class"], "r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    per_class_rows.append(normalize_per_class_row(spec["model"], row))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS_DIR / "model_comparison_metrics_no_t4.csv", metric_rows)
    write_csv(RESULTS_DIR / "model_comparison_per_class_no_t4.csv", per_class_rows)
    with open(RESULTS_DIR / "model_comparison_metrics_no_t4.json", "w", encoding="utf-8") as f:
        json.dump(metric_rows, f, indent=2)

    print("No-T4 model comparison saved.")
    for row in metric_rows:
        print(
            f"{row['model']}: accuracy={float(row['accuracy']):.4f}, "
            f"macro_f1={float(row['macro_f1']):.4f}"
        )


if __name__ == "__main__":
    main()
