#!/usr/bin/env python3
"""
TF-IDF + XGBoost baseline for Java code clone multiclass classification.

This pipeline uses the official train/valid/test splits already present in the
repository, filters classes excluded from the active protocol (T4), and does no
new balancing or split generation.
"""

import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

from clone_experiment_config import EXCLUDED_CLASSES, filter_model_records


SEED = 42
TFIDF_MAX_FEATURES = 1000

random.seed(SEED)
np.random.seed(SEED)


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_pairwise_features(X1_sparse, X2_sparse):
    """
    Constructs a symmetric feature representation:
    [ |v1 - v2|, v1 * v2, cosine_similarity(v1, v2) ]
    """
    t0 = time.time()
    v1 = X1_sparse.toarray()
    v2 = X2_sparse.toarray()

    abs_diff = np.abs(v1 - v2)
    product = v1 * v2

    dot_product = np.sum(v1 * v2, axis=1)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    cosine_sim = dot_product / (norm1 * norm2 + 1e-9)
    cosine_sim = cosine_sim.reshape(-1, 1)

    features = np.hstack([abs_diff, product, cosine_sim]).astype(np.float32)
    print(f"  Feature shape: {features.shape} generated in {time.time() - t0:.2f}s")
    return features


def save_confusion_matrix(path, matrix, class_names):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row.tolist()])


def flatten_report(model_name, report_dict, class_names):
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


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_distribution(split_name, records):
    counts = defaultdict(int)
    for record in records:
        counts[record["clone_type"]] += 1
    dist = ", ".join(f"{label}: {count}" for label, count in sorted(counts.items()))
    print(f"    {split_name} distribution: {dist}")


def main():
    start_time = time.time()
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "model_ready_balanced"
    results_dir = script_dir / "results" / "tfidf_xgboost"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=============================================================")
    print("STARTING TF-IDF + XGBOOST BASELINE PIPELINE")
    print("=============================================================")
    print(f"Data source directory: {data_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Reproducibility Seed: {SEED}")
    print("Using official train/valid/test splits. No new balancing or split generation.")
    print(f"Excluded classes: {sorted(EXCLUDED_CLASSES)}")

    print("\n--- Phase 1: Loading Official Balanced Splits ---")
    train_data_raw = load_jsonl(data_dir / "train_balanced.jsonl")
    valid_data_raw = load_jsonl(data_dir / "valid_balanced.jsonl")
    test_data_raw = load_jsonl(data_dir / "test_balanced.jsonl")

    train_data = filter_model_records(train_data_raw, "Train")
    valid_data = filter_model_records(valid_data_raw, "Valid")
    test_data = filter_model_records(test_data_raw, "Test")

    print(f"  Original train size: {len(train_data_raw):,}")
    print(f"  Original valid size: {len(valid_data_raw):,}")
    print(f"  Original test size: {len(test_data_raw):,}")
    print(f"  Active train size: {len(train_data):,}")
    print(f"  Active valid size: {len(valid_data):,}")
    print(f"  Active test size: {len(test_data):,}")
    print_distribution("Train", train_data)
    print_distribution("Valid", valid_data)
    print_distribution("Test", test_data)

    train_f1 = [record["func1"] for record in train_data]
    train_f2 = [record["func2"] for record in train_data]
    train_labels = [record["clone_type"] for record in train_data]

    valid_f1 = [record["func1"] for record in valid_data]
    valid_f2 = [record["func2"] for record in valid_data]
    valid_labels = [record["clone_type"] for record in valid_data]

    test_f1 = [record["func1"] for record in test_data]
    test_f2 = [record["func2"] for record in test_data]
    test_labels = [record["clone_type"] for record in test_data]

    print("\n--- Phase 2: TF-IDF Feature Extraction ---")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        token_pattern=r"(?u)\b\w+\b",
    )
    vectorizer.fit(train_f1 + train_f2)

    print("  Transforming splits...")
    X1_train = vectorizer.transform(train_f1)
    X2_train = vectorizer.transform(train_f2)
    X1_valid = vectorizer.transform(valid_f1)
    X2_valid = vectorizer.transform(valid_f2)
    X1_test = vectorizer.transform(test_f1)
    X2_test = vectorizer.transform(test_f2)

    print("  Building pairwise features...")
    print("    Train features:")
    X_train = build_pairwise_features(X1_train, X2_train)
    print("    Valid features:")
    X_valid = build_pairwise_features(X1_valid, X2_valid)
    print("    Test features:")
    X_test = build_pairwise_features(X1_test, X2_test)

    print("\n--- Phase 3: Encoding Labels ---")
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_valid = label_encoder.transform(valid_labels)
    y_test = label_encoder.transform(test_labels)
    print(f"  Encoded mapping: {dict(enumerate(label_encoder.classes_))}")

    print("\n--- Phase 4: Training XGBoost ---")
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
    t_train = time.time()
    clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=50)
    print(f"  Training finished in {time.time() - t_train:.2f}s. Best iteration: {clf.best_iteration}")

    print("\n--- Phase 5: Evaluating on Test ---")
    t_predict = time.time()
    y_pred = clf.predict(X_test)
    predict_time = time.time() - t_predict

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
    conf_matrix = confusion_matrix(y_test, y_pred, labels=np.arange(len(label_encoder.classes_)))

    print("\n" + "=" * 60)
    print("DETAILED MULTICLASS CLASSIFICATION REPORT (TEST SPLIT)")
    print("=" * 60)
    print(report)
    print("=" * 60)

    metrics = {
        "model": "TF-IDF + XGBoost",
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "original_train_size": len(train_data_raw),
        "original_valid_size": len(valid_data_raw),
        "original_test_size": len(test_data_raw),
        "train_size": len(train_data),
        "valid_size": len(valid_data),
        "test_size": len(test_data),
        "excluded_classes": sorted(EXCLUDED_CLASSES),
        "active_classes": list(label_encoder.classes_),
        "feature_count": int(X_train.shape[1]),
        "tfidf_max_features": TFIDF_MAX_FEATURES,
        "best_iteration": int(clf.best_iteration),
        "predict_time_seconds": float(predict_time),
        "execution_time_seconds": float(time.time() - start_time),
    }

    print("Saving results and artifacts...")
    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(results_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    save_confusion_matrix(results_dir / "confusion_matrix.csv", conf_matrix, label_encoder.classes_)
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
        json.dump({str(i): c for i, c in enumerate(label_encoder.classes_)}, f, indent=2)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Pipeline finished successfully in {time.time() - start_time:.2f}s.")


if __name__ == "__main__":
    main()
