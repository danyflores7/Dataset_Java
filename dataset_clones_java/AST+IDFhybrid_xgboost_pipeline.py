#!/usr/bin/env python3
"""
Hybrid TF-IDF + AST + XGBoost pipeline for Java clone classification.

This experiment reuses the balanced splits and combines the lexical baseline
features with the structural AST features.
"""

import csv
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

from ast_xgboost_pipeline import (
    SEED,
    extract_split_features as extract_ast_pair_features,
    load_jsonl,
    save_confusion_matrix,
)


TFIDF_MAX_FEATURES = 1000
EPSILON = 1e-9

random.seed(SEED)
np.random.seed(SEED)


BASELINE_SUMMARY = {
    "model": "TF-IDF + XGBoost",
    "accuracy": 0.8454,
    "macro_f1": 0.8730,
    "weighted_f1": 0.8449,
    "feature_count": 2001,
}

BASELINE_PER_CLASS = {
    "MT3": {"precision": 0.5905, "recall": 0.6089, "f1_score": 0.5996, "support": 450},
    "ST3": {"precision": 0.8251, "recall": 0.8489, "f1_score": 0.8368, "support": 450},
    "T0": {"precision": 0.9626, "recall": 0.9733, "f1_score": 0.9680, "support": 450},
    "T1": {"precision": 0.9871, "recall": 1.0000, "f1_score": 0.9935, "support": 229},
    "T2": {"precision": 0.9792, "recall": 0.9038, "f1_score": 0.9400, "support": 52},
    "T4": {"precision": 0.9956, "recall": 0.9978, "f1_score": 0.9967, "support": 450},
    "VST3": {"precision": 0.9143, "recall": 0.9505, "f1_score": 0.9320, "support": 101},
    "WT3": {"precision": 0.7488, "recall": 0.6889, "f1_score": 0.7176, "support": 450},
}

AST_PER_CLASS = {
    "MT3": {"precision": 0.6928, "recall": 0.7467, "f1_score": 0.7187, "support": 450},
    "ST3": {"precision": 0.8592, "recall": 0.9089, "f1_score": 0.8834, "support": 450},
    "T0": {"precision": 0.6330, "recall": 0.5289, "f1_score": 0.5763, "support": 450},
    "T1": {"precision": 0.9784, "recall": 0.7904, "f1_score": 0.8744, "support": 229},
    "T2": {"precision": 0.4300, "recall": 0.8269, "f1_score": 0.5658, "support": 52},
    "T4": {"precision": 0.9758, "recall": 0.9867, "f1_score": 0.9812, "support": 450},
    "VST3": {"precision": 0.8191, "recall": 0.7624, "f1_score": 0.7897, "support": 101},
    "WT3": {"precision": 0.6898, "recall": 0.7067, "f1_score": 0.6981, "support": 450},
}


def build_tfidf_pairwise_features(X1_sparse, X2_sparse) -> np.ndarray:
    """Build baseline-style symmetric TF-IDF pair features."""
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


def extract_tfidf_features(
    train_data: list[dict],
    valid_data: list[dict],
    test_data: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, TfidfVectorizer]:
    """Fit TF-IDF on train only and build symmetric pair features for each split."""
    train_f1 = [record["func1"] for record in train_data]
    train_f2 = [record["func2"] for record in train_data]
    valid_f1 = [record["func1"] for record in valid_data]
    valid_f2 = [record["func2"] for record in valid_data]
    test_f1 = [record["func1"] for record in test_data]
    test_f2 = [record["func2"] for record in test_data]

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        token_pattern=r"(?u)\b\w+\b",
    )
    vectorizer.fit(train_f1 + train_f2)

    X_train = build_tfidf_pairwise_features(
        vectorizer.transform(train_f1),
        vectorizer.transform(train_f2),
    )
    X_valid = build_tfidf_pairwise_features(
        vectorizer.transform(valid_f1),
        vectorizer.transform(valid_f2),
    )
    X_test = build_tfidf_pairwise_features(
        vectorizer.transform(test_f1),
        vectorizer.transform(test_f2),
    )

    return X_train, X_valid, X_test, vectorizer


def load_ast_metrics(results_dir: Path) -> dict:
    """Load AST aggregate metrics when available."""
    ast_metrics_path = results_dir / "ast_xgboost" / "metrics.json"
    if not ast_metrics_path.exists():
        return {
            "model": "AST + XGBoost",
            "accuracy": 0.7773556231003039,
            "macro_f1": 0.7609544601889613,
            "weighted_f1": 0.7771237794223831,
            "feature_count": 58,
        }

    with open(ast_metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_report(
    model_name: str,
    report_dict: dict,
    class_names: np.ndarray,
) -> list[dict]:
    """Convert sklearn classification_report output_dict into graph-friendly rows."""
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


def static_per_class_rows(model_name: str, metrics_by_class: dict) -> list[dict]:
    """Convert documented per-class metrics into graph-friendly rows."""
    rows = []
    for class_name, values in metrics_by_class.items():
        rows.append(
            {
                "model": model_name,
                "class": class_name,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1_score": values["f1_score"],
                "support": values["support"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows as CSV using the keys from the first row."""
    if not rows:
        return

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_comparison_metrics(
    results_dir: Path,
    ast_metrics: dict,
    hybrid_metrics: dict,
    hybrid_report_dict: dict,
    class_names: np.ndarray,
) -> None:
    """Save aggregate and per-class comparison tables for later visualizations."""
    summary_rows = [
        {
            "model": BASELINE_SUMMARY["model"],
            "accuracy": BASELINE_SUMMARY["accuracy"],
            "macro_f1": BASELINE_SUMMARY["macro_f1"],
            "weighted_f1": BASELINE_SUMMARY["weighted_f1"],
            "feature_count": BASELINE_SUMMARY["feature_count"],
            "train_size": hybrid_metrics["train_size"],
            "valid_size": hybrid_metrics["valid_size"],
            "test_size": hybrid_metrics["test_size"],
            "parse_success_rate_train": "",
            "parse_success_rate_valid": "",
            "parse_success_rate_test": "",
        },
        {
            "model": "AST + XGBoost",
            "accuracy": ast_metrics["accuracy"],
            "macro_f1": ast_metrics["macro_f1"],
            "weighted_f1": ast_metrics["weighted_f1"],
            "feature_count": ast_metrics["feature_count"],
            "train_size": ast_metrics.get("train_size", hybrid_metrics["train_size"]),
            "valid_size": ast_metrics.get("valid_size", hybrid_metrics["valid_size"]),
            "test_size": ast_metrics.get("test_size", hybrid_metrics["test_size"]),
            "parse_success_rate_train": ast_metrics.get("parse_success_rate_train", ""),
            "parse_success_rate_valid": ast_metrics.get("parse_success_rate_valid", ""),
            "parse_success_rate_test": ast_metrics.get("parse_success_rate_test", ""),
        },
        {
            "model": hybrid_metrics["model"],
            "accuracy": hybrid_metrics["accuracy"],
            "macro_f1": hybrid_metrics["macro_f1"],
            "weighted_f1": hybrid_metrics["weighted_f1"],
            "feature_count": hybrid_metrics["feature_count"],
            "train_size": hybrid_metrics["train_size"],
            "valid_size": hybrid_metrics["valid_size"],
            "test_size": hybrid_metrics["test_size"],
            "parse_success_rate_train": hybrid_metrics["parse_success_rate_train"],
            "parse_success_rate_valid": hybrid_metrics["parse_success_rate_valid"],
            "parse_success_rate_test": hybrid_metrics["parse_success_rate_test"],
        },
    ]

    per_class_rows = []
    per_class_rows.extend(static_per_class_rows(BASELINE_SUMMARY["model"], BASELINE_PER_CLASS))
    per_class_rows.extend(static_per_class_rows("AST + XGBoost", AST_PER_CLASS))
    per_class_rows.extend(
        flatten_report(hybrid_metrics["model"], hybrid_report_dict, class_names)
    )

    with open(results_dir / "model_comparison_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    write_csv(results_dir / "model_comparison_metrics.csv", summary_rows)

    with open(
        results_dir / "model_comparison_per_class_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(per_class_rows, f, indent=2)
    write_csv(results_dir / "model_comparison_per_class_metrics.csv", per_class_rows)


def main() -> None:
    start_time = time.time()
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "model_ready_balanced"
    results_root = script_dir / "results"
    results_dir = results_root / "hybrid_xgboost"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("STARTING HYBRID TFIDF + AST XGBOOST PIPELINE")
    print(f"Data source directory: {data_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Reproducibility Seed: {SEED}")

    print("Loading balanced splits...")
    train_data = load_jsonl(str(data_dir / "train_balanced.jsonl"))
    valid_data = load_jsonl(str(data_dir / "valid_balanced.jsonl"))
    test_data = load_jsonl(str(data_dir / "test_balanced.jsonl"))

    print(f"Train size: {len(train_data):,}")
    print(f"Valid size: {len(valid_data):,}")
    print(f"Test size: {len(test_data):,}")

    print("Extracting TF-IDF features...")
    X_train_tfidf, X_valid_tfidf, X_test_tfidf, vectorizer = extract_tfidf_features(
        train_data,
        valid_data,
        test_data,
    )

    print("Extracting AST features...")
    X_train_ast, parse_success_rate_train = extract_ast_pair_features(train_data, "Train")
    X_valid_ast, parse_success_rate_valid = extract_ast_pair_features(valid_data, "Valid")
    X_test_ast, parse_success_rate_test = extract_ast_pair_features(test_data, "Test")

    print("Combining hybrid features...")
    X_train = np.hstack([X_train_tfidf, X_train_ast]).astype(np.float32)
    X_valid = np.hstack([X_valid_tfidf, X_valid_ast]).astype(np.float32)
    X_test = np.hstack([X_test_tfidf, X_test_ast]).astype(np.float32)

    print(f"Train TF-IDF shape: {X_train_tfidf.shape}")
    print(f"Train AST shape: {X_train_ast.shape}")
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
        "model": "Hybrid TF-IDF + AST + XGBoost",
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "train_size": len(train_data),
        "valid_size": len(valid_data),
        "test_size": len(test_data),
        "feature_count": int(X_train.shape[1]),
        "tfidf_feature_count": int(X_train_tfidf.shape[1]),
        "ast_feature_count": int(X_train_ast.shape[1]),
        "tfidf_max_features": TFIDF_MAX_FEATURES,
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

    joblib.dump(
        {
            "model": clf,
            "tfidf_vectorizer": vectorizer,
            "label_encoder": label_encoder,
            "tfidf_max_features": TFIDF_MAX_FEATURES,
        },
        results_dir / "model.joblib",
    )

    with open(results_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2)

    save_comparison_metrics(
        results_root,
        load_ast_metrics(results_root),
        metrics,
        report_dict,
        label_encoder.classes_,
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Pipeline finished successfully in {time.time() - start_time:.2f}s.")


if __name__ == "__main__":
    main()
