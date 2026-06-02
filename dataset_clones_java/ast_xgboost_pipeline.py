#!/usr/bin/env python3
"""
AST + XGBoost pipeline for Java code clone multiclass classification.

This experiment reuses the balanced splits produced by the baseline pipeline and
changes only the representation: structural AST features instead of TF-IDF.
"""

import csv
import json
import os
import random
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import javalang
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from clone_experiment_config import EXCLUDED_CLASSES, filter_model_records


SEED = 42
EPSILON = 1e-9

random.seed(SEED)
np.random.seed(SEED)


NODE_COUNT_FEATURES = [
    "n_MethodDeclaration",
    "n_IfStatement",
    "n_ForStatement",
    "n_WhileStatement",
    "n_DoStatement",
    "n_SwitchStatement",
    "n_ReturnStatement",
    "n_MethodInvocation",
    "n_VariableDeclarator",
    "n_LocalVariableDeclaration",
    "n_Assignment",
    "n_BinaryOperation",
    "n_Literal",
    "n_TryStatement",
    "n_CatchClause",
    "n_ThrowStatement",
    "n_ClassCreator",
    "n_MemberReference",
    "n_StatementExpression",
    "n_BlockStatement",
]

AST_FEATURE_NAMES = [
    "parse_success",
    "total_nodes",
    "max_depth",
    *NODE_COUNT_FEATURES,
    "n_parameters",
    "ratio_control_flow",
    "ratio_method_invocations",
    "ratio_assignments",
    "ratio_returns",
]


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL records containing func1, func2, and clone_type."""
    required_fields = {"func1", "func2", "clone_type"}
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.warn(f"{path}:{line_number}: invalid JSON skipped ({exc})")
                continue

            missing = required_fields - set(obj)
            if missing:
                fields = ", ".join(sorted(missing))
                warnings.warn(f"{path}:{line_number}: missing field(s) skipped: {fields}")
                continue

            records.append(
                {
                    "func1": obj["func1"],
                    "func2": obj["func2"],
                    "clone_type": obj["clone_type"],
                }
            )

    return records


def wrap_method_in_class(method_code: str) -> str:
    """Wrap a loose Java method so javalang can parse it as a compilation unit."""
    method_code = (method_code or "").strip()
    return f"public class Dummy {{\n{method_code}\n}}\n"


def parse_java_method(method_code: str) -> Any | None:
    """Parse a Java method wrapped in a dummy class; return None on failure."""
    method_code = (method_code or "").strip()
    wrapped_code = wrap_method_in_class(method_code)

    try:
        return javalang.parse.parse(wrapped_code)
    except Exception:
        return None


def _walk_ast(value: Any, depth: int = 1):
    """Yield javalang AST nodes and their depth."""
    if isinstance(value, javalang.tree.Node):
        yield value, depth
        for child in value.children:
            yield from _walk_ast(child, depth + 1)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _walk_ast(item, depth)


def _empty_ast_features(parse_success: int = 0) -> dict[str, float]:
    features = {name: 0.0 for name in AST_FEATURE_NAMES}
    features["parse_success"] = float(parse_success)
    return features


def extract_ast_features(method_code: str) -> dict[str, float]:
    """Extract numeric AST features from one Java method."""
    tree = parse_java_method(method_code)
    if tree is None:
        return _empty_ast_features(parse_success=0)

    features = _empty_ast_features(parse_success=1)
    total_nodes = 0
    max_depth = 0
    first_method = None

    for node, depth in _walk_ast(tree):
        total_nodes += 1
        max_depth = max(max_depth, depth)

        node_name = type(node).__name__
        count_feature = f"n_{node_name}"
        if count_feature in features:
            features[count_feature] += 1.0

        if first_method is None and isinstance(node, javalang.tree.MethodDeclaration):
            first_method = node

    features["total_nodes"] = float(total_nodes)
    features["max_depth"] = float(max_depth)

    if first_method is not None and first_method.parameters is not None:
        features["n_parameters"] = float(len(first_method.parameters))

    if total_nodes > 0:
        control_flow_nodes = (
            features["n_IfStatement"]
            + features["n_ForStatement"]
            + features["n_WhileStatement"]
            + features["n_DoStatement"]
            + features["n_SwitchStatement"]
        )
        features["ratio_control_flow"] = control_flow_nodes / total_nodes
        features["ratio_method_invocations"] = features["n_MethodInvocation"] / total_nodes
        features["ratio_assignments"] = features["n_Assignment"] / total_nodes
        features["ratio_returns"] = features["n_ReturnStatement"] / total_nodes

    return features


def ast_features_to_vector(features: dict[str, float]) -> np.ndarray:
    """Convert an AST feature dictionary to a stable ordered vector."""
    return np.array([features[name] for name in AST_FEATURE_NAMES], dtype=np.float32)


def build_pairwise_features(vectors_1: np.ndarray, vectors_2: np.ndarray) -> np.ndarray:
    """
    Build an order-invariant pair representation:
    [abs_diff, product, cosine_similarity, euclidean_distance].
    """
    abs_diff = np.abs(vectors_1 - vectors_2)
    product = vectors_1 * vectors_2

    dot_product = np.sum(vectors_1 * vectors_2, axis=1)
    norm_1 = np.linalg.norm(vectors_1, axis=1)
    norm_2 = np.linalg.norm(vectors_2, axis=1)
    cosine_similarity = dot_product / (norm_1 * norm_2 + EPSILON)
    cosine_similarity = cosine_similarity.reshape(-1, 1)

    euclidean_distance = np.linalg.norm(vectors_1 - vectors_2, axis=1).reshape(-1, 1)

    return np.hstack(
        [abs_diff, product, cosine_similarity, euclidean_distance]
    ).astype(np.float32)


def extract_split_features(records: list[dict], split_name: str) -> tuple[np.ndarray, float]:
    """Extract AST features for both functions in each pair and combine them."""
    vectors_1 = []
    vectors_2 = []
    parse_success_values = []

    for record in tqdm(records, desc=f"{split_name} AST features"):
        features_1 = extract_ast_features(record["func1"])
        features_2 = extract_ast_features(record["func2"])

        vectors_1.append(ast_features_to_vector(features_1))
        vectors_2.append(ast_features_to_vector(features_2))
        parse_success_values.extend(
            [features_1["parse_success"], features_2["parse_success"]]
        )

    if not records:
        empty = np.empty((0, (len(AST_FEATURE_NAMES) * 2) + 2), dtype=np.float32)
        return empty, 0.0

    matrix_1 = np.vstack(vectors_1)
    matrix_2 = np.vstack(vectors_2)
    pair_features = build_pairwise_features(matrix_1, matrix_2)
    parse_success_rate = float(np.mean(parse_success_values)) if parse_success_values else 0.0

    return pair_features, parse_success_rate


def save_confusion_matrix(path: Path, matrix: np.ndarray, class_names: np.ndarray) -> None:
    """Save a labeled confusion matrix as CSV."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row.tolist()])


def flatten_report(model_name: str, report_dict: dict, class_names: np.ndarray) -> list[dict]:
    rows = []
    for class_name in class_names:
        values = report_dict[class_name]
        rows.append(
            {
                "model": model_name,
                "class": class_name,
                "precision": float(values["precision"]),
                "recall": float(values["recall"]),
                "f1_score": float(values["f1-score"]),
                "support": int(values["support"]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    start_time = time.time()
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "model_ready_balanced"
    results_dir = script_dir / "results" / "ast_xgboost"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train_balanced.jsonl"
    valid_path = data_dir / "valid_balanced.jsonl"
    test_path = data_dir / "test_balanced.jsonl"

    print("STARTING AST XGBOOST PIPELINE")
    print(f"Data source directory: {data_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Reproducibility Seed: {SEED}")

    print("Loading balanced splits...")
    train_data_raw = load_jsonl(str(train_path))
    valid_data_raw = load_jsonl(str(valid_path))
    test_data_raw = load_jsonl(str(test_path))

    print("Filtering classes excluded from model training/evaluation...")
    train_data = filter_model_records(train_data_raw, "Train")
    valid_data = filter_model_records(valid_data_raw, "Valid")
    test_data = filter_model_records(test_data_raw, "Test")

    print(f"Train size: {len(train_data):,}")
    print(f"Valid size: {len(valid_data):,}")
    print(f"Test size: {len(test_data):,}")

    print("Extracting AST features...")
    X_train, parse_success_rate_train = extract_split_features(train_data, "Train")
    X_valid, parse_success_rate_valid = extract_split_features(valid_data, "Valid")
    X_test, parse_success_rate_test = extract_split_features(test_data, "Test")

    print(f"Train feature shape: {X_train.shape}")
    print(f"Valid feature shape: {X_valid.shape}")
    print(f"Test feature shape: {X_test.shape}")
    print(f"Train parse success rate: {parse_success_rate_train:.4f}")
    print(f"Valid parse success rate: {parse_success_rate_valid:.4f}")
    print(f"Test parse success rate: {parse_success_rate_test:.4f}")

    train_labels = [record["clone_type"] for record in train_data]
    valid_labels = [record["clone_type"] for record in valid_data]
    test_labels = [record["clone_type"] for record in test_data]

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_valid = label_encoder.transform(valid_labels)
    y_test = label_encoder.transform(test_labels)

    label_mapping = {
        str(index): class_name
        for index, class_name in enumerate(label_encoder.classes_)
    }

    print("Training XGBoost...")
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=SEED,
        eval_metric="mlogloss",
        early_stopping_rounds=15,
        tree_method="hist",
        n_jobs=-1,
    )

    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=50,
    )

    print("Evaluating on test...")
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    conf_matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
    )

    print("Saving results...")
    metrics = {
        "model": "AST + XGBoost",
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "train_size": len(train_data),
        "valid_size": len(valid_data),
        "test_size": len(test_data),
        "train_size_original": len(train_data_raw),
        "valid_size_original": len(valid_data_raw),
        "test_size_original": len(test_data_raw),
        "excluded_classes": sorted(EXCLUDED_CLASSES),
        "feature_count": int(X_train.shape[1]),
        "parse_success_rate_train": parse_success_rate_train,
        "parse_success_rate_valid": parse_success_rate_valid,
        "parse_success_rate_test": parse_success_rate_test,
    }

    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(results_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    save_confusion_matrix(
        results_dir / "confusion_matrix.csv",
        conf_matrix,
        label_encoder.classes_,
    )
    write_csv(
        results_dir / "per_class_metrics.csv",
        flatten_report(metrics["model"], report_dict, label_encoder.classes_),
    )

    joblib.dump(clf, results_dir / "model.joblib")

    with open(results_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Pipeline finished successfully in {time.time() - start_time:.2f}s.")


if __name__ == "__main__":
    main()
