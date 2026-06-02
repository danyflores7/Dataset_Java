#!/usr/bin/env python3
"""
Binary product-triage evaluation for the Hybrid TF-IDF + AST + XGBoost model.

The hybrid model is trained as a multiclass classifier. This script does not
retrain it and does not create new splits; it only evaluates existing test data
after dropping classes excluded from all model comparisons.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from ast_xgboost_pipeline import extract_split_features as extract_ast_pair_features
from ast_xgboost_pipeline import load_jsonl
from clone_experiment_config import EXCLUDED_CLASSES, filter_model_records


MODEL_NAME = "Hybrid TF-IDF + AST + XGBoost"
EVALUATION_TASK = "binary_product_triage"
POSITIVE_CLASS = "review_required"
NEGATIVE_CLASS = "no_alert"
NO_ALERT_CLASS = "T0"
REVIEW_CLASSES = {"T1", "T2", "VST3", "ST3", "MT3", "WT3"}
BINARY_LABELS = [NEGATIVE_CLASS, POSITIVE_CLASS]
PER_CLASS_ORDER = ["T1", "T2", "VST3", "ST3", "MT3", "WT3"]
EPSILON = 1e-9


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid multiclass predictions as binary product triage."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=script_dir / "model_ready_balanced",
        help="Directory containing test_balanced.jsonl.",
    )
    parser.add_argument(
        "--hybrid-results-dir",
        type=Path,
        default=script_dir / "results" / "hybrid_xgboost",
        help="Directory containing the trained hybrid model.joblib artifact.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=script_dir / "results" / "hybrid_binary_product",
        help="Directory where binary evaluation outputs will be saved.",
    )
    return parser.parse_args()


def build_tfidf_pairwise_features(X1_sparse, X2_sparse) -> np.ndarray:
    """Build the same symmetric TF-IDF pair features used by the hybrid model."""
    v1 = X1_sparse.toarray().astype(np.float32)
    v2 = X2_sparse.toarray().astype(np.float32)

    abs_diff = np.abs(v1 - v2)
    product = v1 * v2

    dot_product = np.sum(v1 * v2, axis=1)
    norm_1 = np.linalg.norm(v1, axis=1)
    norm_2 = np.linalg.norm(v2, axis=1)
    cosine_similarity = dot_product / (norm_1 * norm_2 + EPSILON)
    cosine_similarity = cosine_similarity.reshape(-1, 1)

    return np.hstack([abs_diff, product, cosine_similarity]).astype(np.float32)


def extract_hybrid_test_features(records: list[dict], model_bundle: dict) -> tuple[np.ndarray, float]:
    """Reconstruct test features with the saved TF-IDF vectorizer and AST extractor."""
    vectorizer = model_bundle["tfidf_vectorizer"]
    test_f1 = [record["func1"] for record in records]
    test_f2 = [record["func2"] for record in records]

    X_test_tfidf = build_tfidf_pairwise_features(
        vectorizer.transform(test_f1),
        vectorizer.transform(test_f2),
    )
    X_test_ast, parse_success_rate_test = extract_ast_pair_features(records, "Test")
    X_test = np.hstack([X_test_tfidf, X_test_ast]).astype(np.float32)

    return X_test, parse_success_rate_test


def class_to_binary_label(class_name: str) -> str | None:
    if class_name == NO_ALERT_CLASS:
        return NEGATIVE_CLASS
    if class_name in REVIEW_CLASSES:
        return POSITIVE_CLASS
    if class_name in EXCLUDED_CLASSES:
        return None
    raise ValueError(f"Unsupported clone class: {class_name}")


def build_prediction_rows(records: list[dict], predicted_classes: np.ndarray) -> list[dict]:
    rows = []
    for index, (record, predicted_multiclass) in enumerate(zip(records, predicted_classes)):
        clone_type = record["clone_type"]
        binary_label = class_to_binary_label(clone_type)
        predicted_binary = class_to_binary_label(str(predicted_multiclass))
        excluded = binary_label is None or predicted_binary is None

        rows.append(
            {
                "index": index,
                "clone_type": clone_type,
                "predicted_multiclass": str(predicted_multiclass),
                "binary_label": binary_label if binary_label is not None else "excluded_class",
                "predicted_binary": predicted_binary if predicted_binary is not None else "excluded_class",
                "excluded_from_main_eval": excluded,
                "correct_binary": "" if excluded else binary_label == predicted_binary,
            }
        )

    return rows


def main_eval_rows(rows: list[dict]) -> list[dict]:
    return [
        row
        for row in rows
        if not row["excluded_from_main_eval"]
        and row["binary_label"] in BINARY_LABELS
        and row["predicted_binary"] in BINARY_LABELS
    ]


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_binary_metrics(rows: list[dict]) -> dict:
    y_true = [row["binary_label"] for row in rows]
    y_pred = [row["predicted_binary"] for row in rows]
    matrix = confusion_matrix(y_true, y_pred, labels=BINARY_LABELS)

    tn = int(matrix[0][0])
    fp = int(matrix[0][1])
    fn = int(matrix[1][0])
    tp = int(matrix[1][1])

    precision_no_alert = safe_divide(tn, tn + fn)
    recall_no_alert = safe_divide(tn, tn + fp)
    f1_no_alert = safe_divide(
        2 * precision_no_alert * recall_no_alert,
        precision_no_alert + recall_no_alert,
    )

    precision_review = safe_divide(tp, tp + fp)
    recall_review = safe_divide(tp, tp + fn)
    f1_review = safe_divide(
        2 * precision_review * recall_review,
        precision_review + recall_review,
    )

    support_no_alert = tn + fp
    support_review = tp + fn
    support = support_no_alert + support_review
    macro_f1 = (f1_no_alert + f1_review) / 2
    weighted_f1 = safe_divide(
        support_no_alert * f1_no_alert + support_review * f1_review,
        support,
    )
    balanced_accuracy = (recall_no_alert + recall_review) / 2

    return {
        "test_accuracy": safe_divide(tn + tp, support),
        "precision_no_alert": precision_no_alert,
        "recall_no_alert": recall_no_alert,
        "f1_no_alert": f1_no_alert,
        "precision_review_required": precision_review,
        "recall_review_required": recall_review,
        "f1_review_required": f1_review,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_accuracy,
        "false_positive_rate": safe_divide(fp, fp + tn),
        "predicted_review_rate": safe_divide(tp + fp, support),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": {
            "labels": BINARY_LABELS,
            "values": matrix.tolist(),
        },
    }


def per_original_class_report(rows: list[dict]) -> list[dict]:
    report_rows = []
    for clone_type in PER_CLASS_ORDER:
        class_rows = [row for row in rows if row["clone_type"] == clone_type]
        support = len(class_rows)
        predicted_review = sum(row["predicted_binary"] == POSITIVE_CLASS for row in class_rows)
        report_rows.append(
            {
                "original_class": clone_type,
                "support": support,
                "predicted_review_required": predicted_review,
                "recall_as_review_required": predicted_review / support if support else 0.0,
            }
        )
    return report_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_confusion_matrix(path: Path, matrix: list[list[int]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted", *BINARY_LABELS])
        for label, row in zip(BINARY_LABELS, matrix):
            writer.writerow([label, *row])


def save_classification_report(path: Path, rows: list[dict]) -> None:
    y_true = [row["binary_label"] for row in rows]
    y_pred = [row["predicted_binary"] for row in rows]
    report = classification_report(
        y_true,
        y_pred,
        labels=BINARY_LABELS,
        target_names=BINARY_LABELS,
        zero_division=0,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)


def print_console_summary(metrics: dict, per_class_rows: list[dict]) -> None:
    print(f"Modelo: {MODEL_NAME}")
    print(f"Evaluation: {EVALUATION_TASK}")
    print(f"Excluded classes: {', '.join(sorted(EXCLUDED_CLASSES))}")
    print(f"Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Recall no_alert: {metrics['recall_no_alert']:.4f}")
    print(f"Recall review_required: {metrics['recall_review_required']:.4f}")
    print(f"False positive rate: {metrics['false_positive_rate']:.4f}")
    print(f"F1 review_required: {metrics['f1_review_required']:.4f}")

    print("\nRecall by original class:")
    for row in per_class_rows:
        label = row["original_class"]
        print(f"  {label}: {row['recall_as_review_required']:.4f}")


def main() -> None:
    start_time = time.time()
    args = parse_args()

    data_dir = args.data_dir.resolve()
    hybrid_results_dir = args.hybrid_results_dir.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    test_path = data_dir / "test_balanced.jsonl"
    model_path = hybrid_results_dir / "model.joblib"

    print("STARTING HYBRID BINARY PRODUCT EVALUATION")
    print(f"Data source directory: {data_dir}")
    print(f"Hybrid model artifact: {model_path}")
    print(f"Results directory: {results_dir}")

    if not model_path.exists():
        raise FileNotFoundError(f"Hybrid model artifact not found: {model_path}")

    print("Loading test split and trained hybrid model...")
    test_data_raw = load_jsonl(str(test_path))
    test_data = filter_model_records(test_data_raw, "Test")
    model_bundle = joblib.load(model_path)
    model = model_bundle["model"]
    label_encoder = model_bundle["label_encoder"]

    print(f"Test size: {len(test_data):,}")
    print("Generating multiclass predictions from existing model...")
    X_test, parse_success_rate_test = extract_hybrid_test_features(test_data, model_bundle)
    y_pred = model.predict(X_test)
    predicted_classes = label_encoder.inverse_transform(y_pred.astype(int))

    prediction_rows = build_prediction_rows(test_data, predicted_classes)
    main_rows = main_eval_rows(prediction_rows)
    metrics_values = compute_binary_metrics(main_rows)
    per_class_rows = per_original_class_report(prediction_rows)

    excluded_prediction_count = len(prediction_rows) - len(main_rows)
    metrics = {
        "model_name": MODEL_NAME,
        "evaluation_task": EVALUATION_TASK,
        "positive_class": POSITIVE_CLASS,
        "negative_class": NEGATIVE_CLASS,
        "excluded_from_main_eval": sorted(EXCLUDED_CLASSES),
        **metrics_values,
        "test_size_original": len(test_data_raw),
        "test_size_used_main_eval": len(main_rows),
        "test_size_excluded_classes": len(test_data_raw) - len(test_data),
        "test_size_excluded_predictions": excluded_prediction_count,
        "parse_success_rate_test": parse_success_rate_test,
        "runtime_seconds": time.time() - start_time,
    }

    write_csv(results_dir / "predictions_test_binary.csv", prediction_rows)
    write_csv(results_dir / "per_original_class_review_recall_test.csv", per_class_rows)
    save_confusion_matrix(
        results_dir / "confusion_matrix.csv",
        metrics_values["confusion_matrix"]["values"],
    )
    save_classification_report(results_dir / "classification_report.txt", main_rows)
    save_json(results_dir / "metrics.json", metrics)

    print_console_summary(metrics, per_class_rows)
    print(f"\nSaved results to: {results_dir}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
