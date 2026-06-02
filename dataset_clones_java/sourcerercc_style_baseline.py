#!/usr/bin/env python3
"""
SourcererCC-style token-overlap baseline for Java clone review triage.

This script is intentionally not a full SourcererCC implementation. The project
already stores candidate pairs as func1/func2 rows, so the repository indexing
step is skipped. We adapt the core idea: normalize Java tokens, compute overlap,
calibrate one threshold on validation, and evaluate once on test.
"""

import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


NO_ALERT_CLASS = "T0"
REVIEW_CLASSES = {"T1", "T2", "VST3", "ST3", "MT3", "WT3"}
CONTEXTUAL_CLASS = "T4"
BINARY_LABELS = ["no_alert", "review_required"]
PER_CLASS_ORDER = ["T1", "T2", "VST3", "ST3", "MT3", "WT3", "T4"]
SELECTION_METRICS = [
    "review_f1",
    "macro_f1",
    "balanced_accuracy",
    "constrained_review_f1",
]
MIN_NO_ALERT_RECALL_FOR_CONSTRAINED = 0.70

JAVA_KEYWORDS = {
    "abstract",
    "assert",
    "boolean",
    "break",
    "byte",
    "case",
    "catch",
    "char",
    "class",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extends",
    "final",
    "finally",
    "float",
    "for",
    "goto",
    "if",
    "implements",
    "import",
    "instanceof",
    "int",
    "interface",
    "long",
    "native",
    "new",
    "package",
    "private",
    "protected",
    "public",
    "return",
    "short",
    "static",
    "strictfp",
    "super",
    "switch",
    "synchronized",
    "this",
    "throw",
    "throws",
    "transient",
    "try",
    "void",
    "volatile",
    "while",
    "true",
    "false",
    "null",
    "var",
    "record",
    "sealed",
    "permits",
    "yield",
}

MULTI_CHAR_TOKENS = [
    ">>>=",
    "<<=",
    ">>=",
    "...",
    ">>>",
    "->",
    "::",
    "++",
    "--",
    "==",
    "!=",
    "<=",
    ">=",
    "&&",
    "||",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "&=",
    "|=",
    "^=",
    "<<",
    ">>",
]
SINGLE_CHAR_TOKENS = set("{}()[];,.?:+-*/%<>=!&|^~@")

NUMBER_RE = re.compile(
    r"""
    (?:
        0[xX][0-9a-fA-F_]+(?:\.[0-9a-fA-F_]*)?(?:[pP][+-]?[0-9_]+)?[fFdDlL]?
      | 0[bB][01_]+[lL]?
      | \d[\d_]*(?:\.(?!\.)[\d_]*)?(?:[eE][+-]?[\d_]+)?[fFdDlL]?
      | \.(?=\d)[\d_]+(?:[eE][+-]?[\d_]+)?[fFdD]?
    )
    """,
    re.VERBOSE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a SourcererCC-style token-overlap baseline."
    )
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=script_dir / "model_ready_balanced",
        help="Directory containing train_balanced.jsonl, valid_balanced.jsonl, and test_balanced.jsonl.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=script_dir / "results" / "sourcerercc_style_baseline",
        help="Directory where baseline outputs will be saved.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional fixed threshold. If omitted, the threshold is calibrated on validation.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=SELECTION_METRICS,
        default="macro_f1",
        help="Metric used to select the validation threshold. Default: macro_f1.",
    )
    parser.add_argument(
        "--include-t4-in-main",
        action="store_true",
        help="Include T4 as review_required in main binary metrics. Default: report T4 separately.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL records containing func1, func2, and clone_type."""
    required_fields = {"func1", "func2", "clone_type"}
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            missing = required_fields - set(obj)
            if missing:
                fields = ", ".join(sorted(missing))
                raise ValueError(f"{path}:{line_number}: missing required field(s): {fields}")
            records.append(
                {
                    "func1": obj["func1"],
                    "func2": obj["func2"],
                    "clone_type": obj["clone_type"],
                }
            )

    return records


def strip_java_comments(code: str) -> str:
    """Remove Java line/block comments while preserving literals."""
    text = code or ""
    out = []
    i = 0
    state = "default"

    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""

        if state == "default":
            if ch == "/" and nxt == "/":
                out.extend("  ")
                i += 2
                state = "line_comment"
                continue
            if ch == "/" and nxt == "*":
                out.extend("  ")
                i += 2
                state = "block_comment"
                continue
            if text.startswith('"""', i):
                out.append('"""')
                i += 3
                state = "text_block"
                continue
            if ch == '"':
                out.append(ch)
                i += 1
                state = "string"
                continue
            if ch == "'":
                out.append(ch)
                i += 1
                state = "char"
                continue
            out.append(ch)
            i += 1
            continue

        if state == "line_comment":
            if ch in "\r\n":
                out.append(ch)
                state = "default"
            else:
                out.append(" ")
            i += 1
            continue

        if state == "block_comment":
            if ch == "*" and nxt == "/":
                out.extend("  ")
                i += 2
                state = "default"
            else:
                out.append(ch if ch in "\r\n" else " ")
                i += 1
            continue

        if state == "string":
            out.append(ch)
            i += 1
            if ch == "\\" and i < len(text):
                out.append(text[i])
                i += 1
            elif ch == '"':
                state = "default"
            continue

        if state == "char":
            out.append(ch)
            i += 1
            if ch == "\\" and i < len(text):
                out.append(text[i])
                i += 1
            elif ch == "'":
                state = "default"
            continue

        if state == "text_block":
            if text.startswith('"""', i):
                out.append('"""')
                i += 3
                state = "default"
            else:
                out.append(ch)
                i += 1

    return "".join(out)


def is_identifier_start(ch: str) -> bool:
    return ch == "_" or ch == "$" or ch.isalpha()


def is_identifier_part(ch: str) -> bool:
    return ch == "_" or ch == "$" or ch.isalnum()


def consume_string(text: str, start: int) -> int:
    i = start + 1
    while i < len(text):
        ch = text[i]
        i += 1
        if ch == "\\" and i < len(text):
            i += 1
        elif ch == '"':
            break
    return i


def consume_char(text: str, start: int) -> int:
    i = start + 1
    while i < len(text):
        ch = text[i]
        i += 1
        if ch == "\\" and i < len(text):
            i += 1
        elif ch == "'":
            break
    return i


def consume_text_block(text: str, start: int) -> int:
    end = text.find('"""', start + 3)
    return len(text) if end == -1 else end + 3


def tokenize_java(code: str) -> list[str]:
    """Tokenize Java and normalize identifiers/literals."""
    text = strip_java_comments(code)
    tokens = []
    i = 0

    while i < len(text):
        ch = text[i]

        if ch.isspace():
            i += 1
            continue

        if text.startswith('"""', i):
            tokens.append("STR_LIT")
            i = consume_text_block(text, i)
            continue

        if ch == '"':
            tokens.append("STR_LIT")
            i = consume_string(text, i)
            continue

        if ch == "'":
            tokens.append("CHAR_LIT")
            i = consume_char(text, i)
            continue

        if ch.isdigit() or (ch == "." and i + 1 < len(text) and text[i + 1].isdigit()):
            match = NUMBER_RE.match(text, i)
            if match:
                tokens.append("NUM_LIT")
                i = match.end()
                continue

        if is_identifier_start(ch):
            j = i + 1
            while j < len(text) and is_identifier_part(text[j]):
                j += 1
            value = text[i:j]
            tokens.append(value if value in JAVA_KEYWORDS else "ID")
            i = j
            continue

        matched = False
        for token in MULTI_CHAR_TOKENS:
            if text.startswith(token, i):
                tokens.append(token)
                i += len(token)
                matched = True
                break
        if matched:
            continue

        if ch in SINGLE_CHAR_TOKENS:
            tokens.append(ch)
        i += 1

    return tokens


def similarity_features(tokens1: list[str], tokens2: list[str]) -> dict:
    counts1 = Counter(tokens1)
    counts2 = Counter(tokens2)
    total1 = sum(counts1.values())
    total2 = sum(counts2.values())

    if total1 == 0 or total2 == 0:
        return {
            "tokens_func1": total1,
            "tokens_func2": total2,
            "shared_token_overlap": 0,
            "overlap_similarity": 0.0,
            "jaccard_similarity": 0.0,
            "containment_1_in_2": 0.0,
            "containment_2_in_1": 0.0,
        }

    shared_tokens = counts1.keys() & counts2.keys()
    overlap = sum(min(counts1[token], counts2[token]) for token in shared_tokens)
    union = total1 + total2 - overlap

    return {
        "tokens_func1": total1,
        "tokens_func2": total2,
        "shared_token_overlap": overlap,
        "overlap_similarity": overlap / min(total1, total2),
        "jaccard_similarity": overlap / union if union else 0.0,
        "containment_1_in_2": overlap / total1,
        "containment_2_in_1": overlap / total2,
    }


def to_binary_target(clone_type: str, include_t4_in_main: bool) -> str | None:
    if clone_type == NO_ALERT_CLASS:
        return "no_alert"
    if clone_type in REVIEW_CLASSES:
        return "review_required"
    if clone_type == CONTEXTUAL_CLASS:
        return "review_required" if include_t4_in_main else None
    raise ValueError(f"Unsupported clone_type: {clone_type}")


def score_split(records: list[dict], split_name: str, include_t4_in_main: bool) -> list[dict]:
    rows = []
    for row_id, record in enumerate(records):
        features = similarity_features(
            tokenize_java(record["func1"]),
            tokenize_java(record["func2"]),
        )
        rows.append(
            {
                "split": split_name,
                "row_id": row_id,
                "clone_type": record["clone_type"],
                "binary_target": to_binary_target(record["clone_type"], include_t4_in_main),
                **features,
            }
        )
    return rows


def predict_label(score: float, threshold: float) -> str:
    return "review_required" if score >= threshold else "no_alert"


def add_predictions(rows: list[dict], threshold: float) -> None:
    for row in rows:
        row["prediction"] = predict_label(row["overlap_similarity"], threshold)


def main_metric_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["binary_target"] is not None]


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def binary_metrics(rows: list[dict]) -> dict:
    y_true = [row["binary_target"] for row in rows]
    y_pred = [row["prediction"] for row in rows]
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
        "support": support,
        "accuracy": float(accuracy_score(y_true, y_pred)),
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


def threshold_search_row(threshold: float, rows: list[dict]) -> dict:
    trial_rows = [
        dict(row, prediction=predict_label(row["overlap_similarity"], threshold))
        for row in rows
    ]
    metrics = binary_metrics(trial_rows)
    return {
        "threshold": threshold,
        "accuracy": metrics["accuracy"],
        "precision_no_alert": metrics["precision_no_alert"],
        "recall_no_alert": metrics["recall_no_alert"],
        "f1_no_alert": metrics["f1_no_alert"],
        "precision_review_required": metrics["precision_review_required"],
        "recall_review_required": metrics["recall_review_required"],
        "f1_review_required": metrics["f1_review_required"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "false_positive_rate": metrics["false_positive_rate"],
        "predicted_review_rate": metrics["predicted_review_rate"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "tn": metrics["tn"],
        "fn": metrics["fn"],
    }


def select_threshold(search_rows: list[dict], selection_metric: str) -> tuple[dict, str]:
    if selection_metric == "review_f1":
        return max(
            search_rows,
            key=lambda row: (
                row["f1_review_required"],
                row["recall_review_required"],
                row["precision_review_required"],
                row["macro_f1"],
                row["threshold"],
            ),
        ), "review_f1"

    if selection_metric == "macro_f1":
        return max(
            search_rows,
            key=lambda row: (
                row["macro_f1"],
                row["balanced_accuracy"],
                row["f1_review_required"],
                row["recall_no_alert"],
                row["threshold"],
            ),
        ), "macro_f1"

    if selection_metric == "balanced_accuracy":
        return max(
            search_rows,
            key=lambda row: (
                row["balanced_accuracy"],
                row["macro_f1"],
                row["f1_review_required"],
                row["recall_no_alert"],
                row["threshold"],
            ),
        ), "balanced_accuracy"

    if selection_metric == "constrained_review_f1":
        constrained_rows = [
            row
            for row in search_rows
            if row["recall_no_alert"] >= MIN_NO_ALERT_RECALL_FOR_CONSTRAINED
        ]
        if not constrained_rows:
            best, _ = select_threshold(search_rows, "macro_f1")
            return best, "macro_f1_fallback_no_threshold_met_no_alert_recall_constraint"

        return max(
            constrained_rows,
            key=lambda row: (
                row["f1_review_required"],
                row["macro_f1"],
                row["balanced_accuracy"],
                row["recall_review_required"],
                row["precision_review_required"],
                row["threshold"],
            ),
        ), "constrained_review_f1"

    raise ValueError(f"Unsupported selection metric: {selection_metric}")


def threshold_search(valid_rows: list[dict], selection_metric: str) -> tuple[float, list[dict], dict, str]:
    search_rows = []
    calibration_rows = main_metric_rows(valid_rows)

    if not calibration_rows:
        raise ValueError("Validation split has no rows for threshold calibration.")

    for i in range(101):
        threshold = i / 100
        search_rows.append(threshold_search_row(threshold, calibration_rows))

    best_row, effective_selection_metric = select_threshold(search_rows, selection_metric)
    return best_row["threshold"], search_rows, best_row, effective_selection_metric


def per_original_class_report(rows: list[dict]) -> list[dict]:
    report_rows = []
    for clone_type in PER_CLASS_ORDER:
        class_rows = [row for row in rows if row["clone_type"] == clone_type]
        support = len(class_rows)
        predicted_review = sum(row["prediction"] == "review_required" for row in class_rows)
        report_rows.append(
            {
                "original_class": clone_type,
                "support": support,
                "predicted_review_required": predicted_review,
                "recall_as_review_required": predicted_review / support if support else 0.0,
            }
        )
    return report_rows


def t4_contextual_report(rows: list[dict]) -> dict:
    t4_rows = [row for row in rows if row["clone_type"] == CONTEXTUAL_CLASS]
    support = len(t4_rows)
    predicted_review = sum(row["prediction"] == "review_required" for row in t4_rows)
    return {
        "support": support,
        "predicted_review_required": predicted_review,
        "review_required_rate": predicted_review / support if support else 0.0,
        "note": "T4 is excluded from the main binary comparison by default and reported contextually.",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict | list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_confusion_matrix(path: Path, matrix: list[list[int]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted", *BINARY_LABELS])
        for label, row in zip(BINARY_LABELS, matrix):
            writer.writerow([label, *row])


def save_classification_report(path: Path, rows: list[dict]) -> None:
    y_true = [row["binary_target"] for row in rows]
    y_pred = [row["prediction"] for row in rows]
    report = classification_report(
        y_true,
        y_pred,
        labels=BINARY_LABELS,
        target_names=BINARY_LABELS,
        zero_division=0,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)


def tokenizer_config() -> dict:
    return {
        "java_comments_removed": True,
        "identifiers_normalized_to": "ID",
        "strings_normalized_to": "STR_LIT",
        "chars_normalized_to": "CHAR_LIT",
        "numbers_normalized_to": "NUM_LIT",
        "keywords_preserved": sorted(JAVA_KEYWORDS),
        "multi_char_tokens_preserved": MULTI_CHAR_TOKENS,
        "single_char_tokens_preserved": sorted(SINGLE_CHAR_TOKENS),
        "similarity": (
            "sum(min(count1[t], count2[t]) for t in shared_tokens) "
            "/ min(total_tokens1, total_tokens2)"
        ),
    }


def main() -> None:
    start_time = time.time()
    args = parse_args()

    data_dir = args.data_dir.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train_balanced.jsonl"
    valid_path = data_dir / "valid_balanced.jsonl"
    test_path = data_dir / "test_balanced.jsonl"

    print("STARTING SOURCERERCC-STYLE TOKEN BASELINE")
    print(f"Data source directory: {data_dir}")
    print(f"Results directory: {results_dir}")
    print("Binary mapping: T0 -> no_alert; T1/T2/VST3/ST3/MT3/WT3 -> review_required")
    print("T4 handling: " + ("included in main metrics" if args.include_t4_in_main else "contextual report only"))
    print(f"Threshold selection metric: {args.selection_metric}")

    print("Loading balanced splits...")
    train_data = load_jsonl(train_path)
    valid_data = load_jsonl(valid_path)
    test_data = load_jsonl(test_path)
    print(f"  Train size: {len(train_data):,}")
    print(f"  Validation size: {len(valid_data):,}")
    print(f"  Test size: {len(test_data):,}")

    print("Scoring token overlap for each split...")
    train_rows = score_split(train_data, "train", args.include_t4_in_main)
    valid_rows = score_split(valid_data, "valid", args.include_t4_in_main)
    test_rows = score_split(test_data, "test", args.include_t4_in_main)

    selected_threshold, search_rows, selected_valid_metrics, effective_selection_metric = threshold_search(
        valid_rows,
        args.selection_metric,
    )
    threshold = selected_threshold if args.threshold is None else args.threshold
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("--threshold must be between 0.0 and 1.0")
    threshold_source = "validation_selection" if args.threshold is None else "manual_override"
    best_threshold_metrics_valid = (
        selected_valid_metrics
        if args.threshold is None
        else threshold_search_row(threshold, main_metric_rows(valid_rows))
    )

    print(f"Effective selection metric: {effective_selection_metric}")
    print(f"Best overlap threshold: {threshold:.2f}")
    print("Best threshold metrics on validation:")
    print(f"  Recall no_alert: {best_threshold_metrics_valid['recall_no_alert']:.4f}")
    print(f"  Recall review_required: {best_threshold_metrics_valid['recall_review_required']:.4f}")
    print(f"  False positive rate: {best_threshold_metrics_valid['false_positive_rate']:.4f}")
    print(f"  F1 review_required: {best_threshold_metrics_valid['f1_review_required']:.4f}")
    print(f"  Macro F1: {best_threshold_metrics_valid['macro_f1']:.4f}")
    print(f"  Balanced accuracy: {best_threshold_metrics_valid['balanced_accuracy']:.4f}")

    for rows in (train_rows, valid_rows, test_rows):
        add_predictions(rows, threshold)

    valid_main_rows = main_metric_rows(valid_rows)
    test_main_rows = main_metric_rows(test_rows)
    valid_metrics = binary_metrics(valid_main_rows)
    test_metrics = binary_metrics(test_main_rows)

    metrics = {
        "model": "SourcererCC-style token overlap baseline",
        "selection_metric": args.selection_metric,
        "effective_selection_metric": effective_selection_metric,
        "best_threshold": threshold,
        "threshold_source": threshold_source,
        "best_threshold_metrics_valid": best_threshold_metrics_valid,
        "test_main_metrics": test_metrics,
        "t4_excluded_from_main_eval": not bool(args.include_t4_in_main),
        "tokenizer_config": tokenizer_config(),
        "train_size": len(train_data),
        "valid_size": len(valid_data),
        "test_size": len(test_data),
        "include_t4_in_main": bool(args.include_t4_in_main),
        "selected_threshold_by_validation": selected_threshold,
        "valid_main_metrics": valid_metrics,
        "runtime_seconds": time.time() - start_time,
    }

    test_confusion = test_metrics["confusion_matrix"]["values"]
    per_class_rows = per_original_class_report(test_rows)
    t4_report = t4_contextual_report(test_rows)

    write_csv(results_dir / "threshold_search_valid.csv", search_rows)
    write_csv(results_dir / "predictions_train.csv", train_rows)
    write_csv(results_dir / "predictions_valid.csv", valid_rows)
    write_csv(results_dir / "predictions_test.csv", test_rows)
    write_csv(results_dir / "per_original_class_review_recall_test.csv", per_class_rows)
    save_confusion_matrix(results_dir / "confusion_matrix.csv", test_confusion)
    save_classification_report(results_dir / "classification_report.txt", test_main_rows)
    save_json(results_dir / "metrics.json", metrics)
    save_json(results_dir / "t4_contextual_test_report.json", t4_report)

    print("Test metrics (main binary comparison):")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Recall no_alert: {test_metrics['recall_no_alert']:.4f}")
    print(f"  Precision review_required: {test_metrics['precision_review_required']:.4f}")
    print(f"  Recall review_required: {test_metrics['recall_review_required']:.4f}")
    print(f"  False positive rate: {test_metrics['false_positive_rate']:.4f}")
    print(f"  F1 review_required: {test_metrics['f1_review_required']:.4f}")
    print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {test_metrics['weighted_f1']:.4f}")
    print(f"  Balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Saved results to: {results_dir}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
