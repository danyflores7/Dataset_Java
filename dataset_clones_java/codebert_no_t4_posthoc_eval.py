#!/usr/bin/env python3
"""
Post-hoc no-T4 evaluation for pulled CodeBERT + XGBoost results.

This does not retrain CodeBERT. It reads the existing multiclass confusion
matrix, removes T4 as an actual/ground-truth class, and keeps predicted T4 as
an error for active classes. The output is explicitly marked as post-hoc so it
is not confused with a real no-T4 retraining run.
"""

import csv
import json
from pathlib import Path

from clone_experiment_config import EXCLUDED_CLASSES, MODEL_CLASSES, to_binary_product_label


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results" / "codebert_xgboost"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.csv"

BINARY_LABELS = ["no_alert", "review_required"]
REVIEW_REQUIRED = "review_required"


def safe_div(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def read_confusion_matrix(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        pred_labels = header[1:]
        rows = []
        actual_labels = []
        for row in reader:
            actual_labels.append(row[0])
            rows.append([int(value) for value in row[1:]])
    return actual_labels, pred_labels, rows


def write_confusion_matrix(path, actual_labels, pred_labels, matrix):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted", *pred_labels])
        for label, row in zip(actual_labels, matrix):
            writer.writerow([label, *row])


def class_metrics_from_confusion(actual_labels, pred_labels, matrix, active_labels):
    actual_idx = {label: i for i, label in enumerate(actual_labels)}
    pred_idx = {label: i for i, label in enumerate(pred_labels)}
    active_actual_indices = [actual_idx[label] for label in active_labels]

    rows = []
    total_support = 0
    total_correct = 0

    for label in active_labels:
        row_index = actual_idx[label]
        col_index = pred_idx[label]
        support = sum(matrix[row_index])
        tp = matrix[row_index][col_index]
        fp = sum(matrix[i][col_index] for i in active_actual_indices if i != row_index)
        fn = support - tp
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        rows.append(
            {
                "class": label,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": support,
            }
        )
        total_support += support
        total_correct += tp

    accuracy = safe_div(total_correct, total_support)
    macro_f1 = safe_div(sum(row["f1_score"] for row in rows), len(rows))
    weighted_f1 = safe_div(
        sum(row["f1_score"] * row["support"] for row in rows),
        total_support,
    )

    return rows, accuracy, macro_f1, weighted_f1, total_support, total_correct


def binary_metrics_from_confusion(actual_labels, pred_labels, matrix, active_labels):
    actual_idx = {label: i for i, label in enumerate(actual_labels)}
    counts = {
        ("no_alert", "no_alert"): 0,
        ("no_alert", "review_required"): 0,
        ("review_required", "no_alert"): 0,
        ("review_required", "review_required"): 0,
    }

    for actual_label in active_labels:
        actual_binary = to_binary_product_label(actual_label)
        row = matrix[actual_idx[actual_label]]
        for pred_label, count in zip(pred_labels, row):
            pred_binary = "no_alert" if pred_label == "T0" else "review_required"
            counts[(actual_binary, pred_binary)] += count

    tn = counts[("no_alert", "no_alert")]
    fp = counts[("no_alert", "review_required")]
    fn = counts[("review_required", "no_alert")]
    tp = counts[("review_required", "review_required")]

    precision_no_alert = safe_div(tn, tn + fn)
    recall_no_alert = safe_div(tn, tn + fp)
    f1_no_alert = safe_div(2 * precision_no_alert * recall_no_alert, precision_no_alert + recall_no_alert)

    precision_review = safe_div(tp, tp + fp)
    recall_review = safe_div(tp, tp + fn)
    f1_review = safe_div(2 * precision_review * recall_review, precision_review + recall_review)

    total = tn + fp + fn + tp
    accuracy = safe_div(tn + tp, total)
    macro_f1 = (f1_no_alert + f1_review) / 2
    weighted_f1 = safe_div(
        f1_no_alert * (tn + fp) + f1_review * (tp + fn),
        total,
    )
    balanced_accuracy = (recall_no_alert + recall_review) / 2

    return {
        "evaluation": "binary_product_view_posthoc_no_t4",
        "posthoc_filter": True,
        "retrained_without_t4": False,
        "labels": BINARY_LABELS,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_accuracy,
        "precision_no_alert": precision_no_alert,
        "recall_no_alert": recall_no_alert,
        "f1_no_alert": f1_no_alert,
        "precision_review_required": precision_review,
        "recall_review_required": recall_review,
        "f1_review_required": f1_review,
        "false_positive_rate": safe_div(fp, fp + tn),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def write_class_metrics_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1_score", "support"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_binary_per_class_csv(path, actual_labels, pred_labels, matrix, active_labels):
    actual_idx = {label: i for i, label in enumerate(actual_labels)}
    rows = []
    for label in active_labels:
        row = matrix[actual_idx[label]]
        support = sum(row)
        predicted_review = sum(
            count for pred_label, count in zip(pred_labels, row) if pred_label != "T0"
        )
        rows.append(
            {
                "original_class": label,
                "support": support,
                "predicted_review_required": predicted_review,
                "recall_as_review_required": safe_div(predicted_review, support),
            }
        )

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "original_class",
                "support",
                "predicted_review_required",
                "recall_as_review_required",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, rows, accuracy, macro_f1, weighted_f1, total_support):
    with open(path, "w", encoding="utf-8") as f:
        f.write("CodeBERT + XGBoost post-hoc no-T4 evaluation\n")
        f.write("T4 ground-truth rows are excluded. Predicted T4 remains an error.\n\n")
        f.write(f"{'class':<12}{'precision':>11}{'recall':>11}{'f1-score':>11}{'support':>10}\n")
        for row in rows:
            f.write(
                f"{row['class']:<12}"
                f"{row['precision']:>11.4f}"
                f"{row['recall']:>11.4f}"
                f"{row['f1_score']:>11.4f}"
                f"{row['support']:>10}\n"
            )
        f.write("\n")
        f.write(f"{'accuracy':<12}{'':>11}{'':>11}{accuracy:>11.4f}{total_support:>10}\n")
        f.write(f"{'macro avg':<12}{'':>11}{'':>11}{macro_f1:>11.4f}{total_support:>10}\n")
        f.write(f"{'weighted avg':<12}{'':>11}{'':>11}{weighted_f1:>11.4f}{total_support:>10}\n")


def main():
    actual_labels, pred_labels, matrix = read_confusion_matrix(CONFUSION_MATRIX_PATH)
    active_labels = [label for label in MODEL_CLASSES if label in actual_labels and label not in EXCLUDED_CLASSES]
    active_row_indices = [actual_labels.index(label) for label in active_labels]
    excluded_row_indices = [i for i, label in enumerate(actual_labels) if label in EXCLUDED_CLASSES]

    active_matrix = [matrix[i] for i in active_row_indices]
    excluded_actual_support = sum(sum(matrix[i]) for i in excluded_row_indices)
    predicted_excluded_for_active = sum(
        row[pred_labels.index(label)]
        for row in active_matrix
        for label in EXCLUDED_CLASSES
        if label in pred_labels
    )

    rows, accuracy, macro_f1, weighted_f1, total_support, total_correct = class_metrics_from_confusion(
        actual_labels,
        pred_labels,
        matrix,
        active_labels,
    )
    binary_metrics = binary_metrics_from_confusion(actual_labels, pred_labels, matrix, active_labels)

    metrics = {
        "model": "CodeBERT + XGBoost",
        "evaluation": "multiclass_posthoc_no_t4",
        "posthoc_filter": True,
        "retrained_without_t4": False,
        "excluded_actual_classes": sorted(EXCLUDED_CLASSES),
        "active_classes": active_labels,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "test_size": total_support,
        "correct_predictions": total_correct,
        "excluded_actual_support": excluded_actual_support,
        "predicted_excluded_class_for_active_rows": predicted_excluded_for_active,
        "source_confusion_matrix": str(CONFUSION_MATRIX_PATH),
        "binary_product_metrics_file": "binary_product_metrics_no_t4_eval.json",
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "metrics_no_t4_eval.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(RESULTS_DIR / "binary_product_metrics_no_t4_eval.json", "w", encoding="utf-8") as f:
        json.dump(binary_metrics, f, indent=2)

    write_class_metrics_csv(RESULTS_DIR / "per_class_metrics_no_t4_eval.csv", rows)
    write_report(
        RESULTS_DIR / "classification_report_no_t4_eval.txt",
        rows,
        accuracy,
        macro_f1,
        weighted_f1,
        total_support,
    )
    write_confusion_matrix(
        RESULTS_DIR / "confusion_matrix_no_t4_eval.csv",
        active_labels,
        pred_labels,
        active_matrix,
    )
    write_confusion_matrix(
        RESULTS_DIR / "binary_product_confusion_matrix_no_t4_eval.csv",
        BINARY_LABELS,
        BINARY_LABELS,
        [
            [binary_metrics["tn"], binary_metrics["fp"]],
            [binary_metrics["fn"], binary_metrics["tp"]],
        ],
    )
    write_binary_per_class_csv(
        RESULTS_DIR / "binary_product_per_original_class_review_recall_no_t4_eval.csv",
        actual_labels,
        pred_labels,
        matrix,
        active_labels,
    )

    print("CodeBERT post-hoc no-T4 evaluation saved.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Binary macro F1: {binary_metrics['macro_f1']:.4f}")
    print(f"Binary balanced accuracy: {binary_metrics['balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()
