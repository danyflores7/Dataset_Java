#!/usr/bin/env python3
"""
CodeBERT + XGBoost Pipeline for Java Code Clone Multiclass Classification

This script implements Option A:
1. Loads the pre-existing balanced dataset splits (Train/Valid/Test).
2. Detects hardware acceleration (Apple Silicon MPS, CUDA, or CPU).
3. Downloads and loads microsoft/codebert-base from Hugging Face.
4. Extracts semantic embeddings (CLS token, 768 dimensions) in batches.
5. Caches the extracted embeddings to disk (.npy) for instant subsequent runs.
6. Computes pairwise symmetric features:
   - Absolute difference: |e1 - e2| (768 features)
   - Element-wise product: e1 * e2 (768 features)
   - Cosine similarity: (e1 . e2) / (||e1|| * ||e2||) (1 feature)
   - Total features: 1,537.
7. Trains a multiclass XGBoost classifier with early stopping on validation.
8. Evaluates on the test split and outputs a detailed classification report.
"""

import os
import sys
import json
import csv
import argparse
import time
import random
import numpy as np
import xgboost as xgb
import torch
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
import joblib

# Ensure we can import from local path
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir))

from ast_xgboost_pipeline import load_jsonl, SEED, save_confusion_matrix
from clone_experiment_config import (
    EXCLUDED_CLASSES,
    MODEL_CLASSES,
    filter_model_records,
    to_binary_product_label,
)

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Configurations
MODEL_NAME = "microsoft/codebert-base"
BATCH_SIZE = 32
MAX_LENGTH = 512  # CodeBERT context length limit
BINARY_LABELS = ["no_alert", "review_required"]
DEFAULT_ENCODED_PAIRS_PATH = script_dir.parent.parent / "encoded_pairs.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CodeBERT + XGBoost on official splits with T4 excluded."
    )
    parser.add_argument(
        "--encoded-pairs-path",
        type=Path,
        default=None,
        help=(
            "Optional JSONL with precomputed CodeBERT pair features. Use only when "
            "the file covers the official train/valid/test rows and can be joined "
            "to them by id1/id2."
        ),
    )
    return parser.parse_args()

def select_device():
    """Detect and return the best available torch device."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  Hardware Acceleration: Apple Silicon GPU (MPS) detected!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("  Hardware Acceleration: NVIDIA GPU (CUDA) detected!")
    else:
        device = torch.device("cpu")
        print("  Hardware Acceleration: None (using CPU).")
    return device

def extract_codebert_embeddings(records, tokenizer, model, device, desc="Extracting"):
    """
    Extract CodeBERT CLS embeddings (768 dimensions) for both functions in each pair.
    Returns: (embeddings_1, embeddings_2) as numpy arrays.
    """
    model.eval()
    
    # Process both func1 and func2
    embeddings_1 = []
    embeddings_2 = []
    
    # Extract list of texts
    funcs_1 = [record["func1"] for record in records]
    funcs_2 = [record["func2"] for record in records]
    
    def process_list(text_list, name):
        all_embeddings = []
        # Process in batches
        for i in tqdm(range(0, len(text_list), BATCH_SIZE), desc=f"{desc} {name}"):
            batch_texts = text_list[i : i + BATCH_SIZE]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                # CLS token representation is the first token (index 0)
                # shape: (batch_size, 768)
                cls_repr = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(cls_repr)
                
        return np.vstack(all_embeddings)
        
    emb1 = process_list(funcs_1, "func1")
    emb2 = process_list(funcs_2, "func2")
    return emb1, emb2

def get_or_load_embeddings(
    split_name,
    records,
    data_dir,
    cache_dir,
    tokenizer,
    model,
    device,
    legacy_cache_dir=None,
    active_indices=None,
    raw_count=None,
):
    """Load cached embeddings from disk if available, otherwise calculate and cache them."""
    emb1_path = cache_dir / f"{split_name}_emb1.npy"
    emb2_path = cache_dir / f"{split_name}_emb2.npy"
    
    if emb1_path.exists() and emb2_path.exists():
        print(f"  Found cached embeddings for {split_name} split. Loading...")
        emb1 = np.load(emb1_path)
        emb2 = np.load(emb2_path)
        if len(emb1) != len(records) or len(emb2) != len(records):
            raise ValueError(
                f"Cached embeddings for {split_name} have {len(emb1)} / {len(emb2)} rows, "
                f"but the active split has {len(records)} rows. Delete {cache_dir} and rerun."
            )
        return emb1, emb2

    if legacy_cache_dir is not None and active_indices is not None:
        legacy_emb1_path = legacy_cache_dir / f"{split_name}_emb1.npy"
        legacy_emb2_path = legacy_cache_dir / f"{split_name}_emb2.npy"
        if legacy_emb1_path.exists() and legacy_emb2_path.exists():
            print(f"  Found legacy cached embeddings for {split_name}. Filtering excluded classes...")
            legacy_emb1 = np.load(legacy_emb1_path)
            legacy_emb2 = np.load(legacy_emb2_path)
            expected_raw_count = raw_count if raw_count is not None else len(legacy_emb1)
            if len(legacy_emb1) == expected_raw_count and len(legacy_emb2) == expected_raw_count:
                emb1 = legacy_emb1[active_indices]
                emb2 = legacy_emb2[active_indices]
                np.save(emb1_path, emb1)
                np.save(emb2_path, emb2)
                return emb1, emb2
            print(
                f"  Legacy cache row count mismatch for {split_name}: "
                f"{len(legacy_emb1)} / {len(legacy_emb2)} vs expected {expected_raw_count}. Re-extracting."
            )
        
    print(f"  No cache found for {split_name} split. Extracting from CodeBERT...")
    t_start = time.time()
    emb1, emb2 = extract_codebert_embeddings(records, tokenizer, model, device, desc=split_name)
    print(f"  Extracted in {time.time() - t_start:.2f}s. Saving to cache...")
    
    # Save to disk
    np.save(emb1_path, emb1)
    np.save(emb2_path, emb2)
    return emb1, emb2

def build_pairwise_features(emb1, emb2):
    """
    Constructs order-invariant pair features:
    [ |e1 - e2|, e1 * e2, cosine_similarity(e1, e2) ]
    """
    abs_diff = np.abs(emb1 - emb2)
    product = emb1 * emb2
    
    dot_product = np.sum(emb1 * emb2, axis=1)
    norm1 = np.linalg.norm(emb1, axis=1)
    norm2 = np.linalg.norm(emb2, axis=1)
    epsilon = 1e-9
    cosine_sim = dot_product / (norm1 * norm2 + epsilon)
    cosine_sim = cosine_sim.reshape(-1, 1)
    
    return np.hstack([abs_diff, product, cosine_sim]).astype(np.float32)

def get_pair_key(record):
    """Return the split/embedding join key for a code pair."""
    try:
        return str(record["id1"]), str(record["id2"])
    except KeyError as exc:
        raise KeyError("Records must include id1 and id2 to join precomputed CodeBERT embeddings.") from exc

def feature_vector_from_encoded_row(row):
    """Build the same symmetric pair feature vector used by the local CodeBERT path."""
    if "abs_diff" in row and "prod" in row and "cosine" in row:
        abs_diff = np.asarray(row["abs_diff"], dtype=np.float32)
        product = np.asarray(row["prod"], dtype=np.float32)
        cosine = np.asarray([row["cosine"]], dtype=np.float32)
        return np.concatenate([abs_diff, product, cosine])

    if "u" not in row or "v" not in row:
        raise ValueError("Encoded row must contain abs_diff/prod/cosine or u/v embeddings.")

    u = np.asarray(row["u"], dtype=np.float32)
    v = np.asarray(row["v"], dtype=np.float32)
    abs_diff = np.abs(u - v)
    product = u * v
    denominator = np.linalg.norm(u) * np.linalg.norm(v) + 1e-9
    cosine = np.asarray([float(np.dot(u, v) / denominator)], dtype=np.float32)
    return np.concatenate([abs_diff, product, cosine])

def load_encoded_pair_index(encoded_pairs_path):
    """Load precomputed CodeBERT pair embeddings keyed by (id1, id2)."""
    index = {}
    duplicate_count = 0
    skipped_excluded = 0

    with open(encoded_pairs_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            clone_type = row.get("clone_type")
            if clone_type in EXCLUDED_CLASSES:
                skipped_excluded += 1
                continue

            key = get_pair_key(row)
            if key in index:
                duplicate_count += 1
                continue

            index[key] = feature_vector_from_encoded_row(row)

    print(f"  Encoded active pairs loaded: {len(index):,}")
    print(f"  Encoded pairs skipped by excluded classes: {skipped_excluded:,}")
    if duplicate_count:
        print(f"  Duplicate encoded pair keys ignored: {duplicate_count:,}")

    return index

def build_features_from_encoded_pairs(split_name, records, encoded_index):
    """Build a feature matrix for one official split by joining on id1/id2."""
    features = []
    missing = []

    for record in records:
        key = get_pair_key(record)
        encoded_features = encoded_index.get(key)
        if encoded_features is None:
            # The feature construction is symmetric, so reversed IDs are equivalent here.
            encoded_features = encoded_index.get((key[1], key[0]))

        if encoded_features is None:
            missing.append(key)
            continue

        features.append(encoded_features)

    if missing:
        examples = ", ".join(f"{id1}/{id2}" for id1, id2 in missing[:5])
        raise ValueError(
            f"Missing CodeBERT embeddings for {len(missing):,} {split_name} rows. "
            f"Examples: {examples}. Regenerate encoded_pairs.jsonl for the official splits."
        )

    X = np.vstack(features).astype(np.float32)
    print(f"  {split_name} feature shape from encoded pairs: {X.shape}")
    return X

def save_binary_product_outputs(results_dir, test_labels, pred_labels):
    """Save the product-oriented binary view derived from multiclass predictions."""
    y_true_binary = [to_binary_product_label(label) for label in test_labels]
    y_pred_binary = [to_binary_product_label(label) for label in pred_labels]

    binary_conf = confusion_matrix(
        y_true_binary,
        y_pred_binary,
        labels=BINARY_LABELS,
    )
    tn, fp, fn, tp = binary_conf.ravel()

    binary_report = classification_report(
        y_true_binary,
        y_pred_binary,
        labels=BINARY_LABELS,
        target_names=BINARY_LABELS,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "evaluation": "binary_product_view_from_multiclass_predictions",
        "labels": BINARY_LABELS,
        "excluded_classes": sorted(EXCLUDED_CLASSES),
        "accuracy": float(accuracy_score(y_true_binary, y_pred_binary)),
        "macro_f1": float(f1_score(y_true_binary, y_pred_binary, average="macro")),
        "weighted_f1": float(f1_score(y_true_binary, y_pred_binary, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_binary, y_pred_binary)),
        "precision_no_alert": float(binary_report["no_alert"]["precision"]),
        "recall_no_alert": float(binary_report["no_alert"]["recall"]),
        "f1_no_alert": float(binary_report["no_alert"]["f1-score"]),
        "precision_review_required": float(binary_report["review_required"]["precision"]),
        "recall_review_required": float(binary_report["review_required"]["recall"]),
        "f1_review_required": float(binary_report["review_required"]["f1-score"]),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    with open(results_dir / "binary_product_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_confusion_matrix(
        results_dir / "binary_product_confusion_matrix.csv",
        binary_conf,
        BINARY_LABELS,
    )

    review_required = "review_required"
    rows = []
    for class_name in MODEL_CLASSES:
        class_indices = [i for i, label in enumerate(test_labels) if label == class_name]
        if not class_indices:
            continue
        predicted_review = sum(1 for i in class_indices if y_pred_binary[i] == review_required)
        rows.append({
            "original_class": class_name,
            "support": len(class_indices),
            "predicted_review_required": predicted_review,
            "recall_as_review_required": predicted_review / len(class_indices),
        })

    with open(results_dir / "binary_product_per_original_class_review_recall.csv", "w", encoding="utf-8", newline="") as f:
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
        for row in rows:
            writer.writerow(row)

    return metrics

def main():
    args = parse_args()
    start_time = time.time()
    data_dir = script_dir / "model_ready_balanced"
    results_dir = script_dir / "results" / "codebert_xgboost"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    encoded_pairs_path = args.encoded_pairs_path

    # Cache directory for numpy embeddings after applying the active class policy.
    cache_dir = results_dir / "cache_no_t4"
    cache_dir.mkdir(parents=True, exist_ok=True)
    legacy_cache_dir = results_dir / "cache"
    
    print("=============================================================")
    print("STARTING CODEBERT + XGBOOST SEMANTIC CLASSIFICATION PIPELINE")
    print("=============================================================")
    print(f"Data source directory: {data_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Reproducibility Seed: {SEED}")
    
    # 1. Load balanced datasets
    print("Loading balanced splits...")
    train_data_raw = load_jsonl(str(data_dir / "train_balanced.jsonl"))
    valid_data_raw = load_jsonl(str(data_dir / "valid_balanced.jsonl"))
    test_data_raw = load_jsonl(str(data_dir / "test_balanced.jsonl"))

    train_active_indices = np.array(
        [i for i, record in enumerate(train_data_raw) if record["clone_type"] not in EXCLUDED_CLASSES],
        dtype=np.int64,
    )
    valid_active_indices = np.array(
        [i for i, record in enumerate(valid_data_raw) if record["clone_type"] not in EXCLUDED_CLASSES],
        dtype=np.int64,
    )
    test_active_indices = np.array(
        [i for i, record in enumerate(test_data_raw) if record["clone_type"] not in EXCLUDED_CLASSES],
        dtype=np.int64,
    )

    train_data = filter_model_records(train_data_raw, "Train")
    valid_data = filter_model_records(valid_data_raw, "Valid")
    test_data = filter_model_records(test_data_raw, "Test")
    
    print(f"  Original train size: {len(train_data_raw):,}")
    print(f"  Original valid size: {len(valid_data_raw):,}")
    print(f"  Original test size: {len(test_data_raw):,}")
    print(f"  Active train size: {len(train_data):,}")
    print(f"  Active valid size: {len(valid_data):,}")
    print(f"  Active test size: {len(test_data):,}")
    print(f"  Excluded classes: {', '.join(sorted(EXCLUDED_CLASSES))}")
    
    device = "precomputed_encoded_pairs"

    if encoded_pairs_path is not None:
        if not encoded_pairs_path.exists():
            raise FileNotFoundError(f"Encoded pairs file not found: {encoded_pairs_path}")

        print("\n--- Phase 1: Loading Precomputed CodeBERT Pair Embeddings ---")
        print(f"  Encoded pairs path: {encoded_pairs_path}")
        encoded_index = load_encoded_pair_index(encoded_pairs_path)

        print("\n--- Phase 2: Joining Encoded Features to Official Splits ---")
        X_train = build_features_from_encoded_pairs("Train", train_data, encoded_index)
        X_valid = build_features_from_encoded_pairs("Valid", valid_data, encoded_index)
        X_test = build_features_from_encoded_pairs("Test", test_data, encoded_index)
    else:
        # Check if cache exists for all splits to avoid loading Hugging Face model
        cache_exists = all(
            (
                (cache_dir / f"{split}_emb1.npy").exists()
                and (cache_dir / f"{split}_emb2.npy").exists()
            )
            or (
                (legacy_cache_dir / f"{split}_emb1.npy").exists()
                and (legacy_cache_dir / f"{split}_emb2.npy").exists()
            )
            for split in ["train", "valid", "test"]
        )

        tokenizer = None
        model = None

        if not cache_exists:
            print("\n--- Loading Pre-trained CodeBERT ---")
            device = select_device()

            # Suppress huggingface download warning output
            from transformers import AutoTokenizer, AutoModel
            print(f"  Loading model and tokenizer: {MODEL_NAME}...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModel.from_pretrained(MODEL_NAME).to(device)
            print("  Model loaded successfully.")
        else:
            device = "cached_local_embeddings"

        print("\n--- Phase 1: Extracting Semantic Embeddings ---")
        train_emb1, train_emb2 = get_or_load_embeddings(
            "train",
            train_data,
            data_dir,
            cache_dir,
            tokenizer,
            model,
            device,
            legacy_cache_dir=legacy_cache_dir,
            active_indices=train_active_indices,
            raw_count=len(train_data_raw),
        )
        valid_emb1, valid_emb2 = get_or_load_embeddings(
            "valid",
            valid_data,
            data_dir,
            cache_dir,
            tokenizer,
            model,
            device,
            legacy_cache_dir=legacy_cache_dir,
            active_indices=valid_active_indices,
            raw_count=len(valid_data_raw),
        )
        test_emb1, test_emb2 = get_or_load_embeddings(
            "test",
            test_data,
            data_dir,
            cache_dir,
            tokenizer,
            model,
            device,
            legacy_cache_dir=legacy_cache_dir,
            active_indices=test_active_indices,
            raw_count=len(test_data_raw),
        )

        print("\n--- Phase 2: Generating Symmetric Pair Features ---")
        X_train = build_pairwise_features(train_emb1, train_emb2)
        X_valid = build_pairwise_features(valid_emb1, valid_emb2)
        X_test = build_pairwise_features(test_emb1, test_emb2)

        print(f"  Train feature shape: {X_train.shape}")
        print(f"  Valid feature shape: {X_valid.shape}")
        print(f"  Test feature shape: {X_test.shape}")
    
    # 2. Encode Multi-class Target Labels
    print("\n--- Phase 3: Encoding Multi-class Labels ---")
    train_labels = [record["clone_type"] for record in train_data]
    valid_labels = [record["clone_type"] for record in valid_data]
    test_labels = [record["clone_type"] for record in test_data]
    
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_valid = label_encoder.transform(valid_labels)
    y_test = label_encoder.transform(test_labels)
    
    label_mapping = {str(index): class_name for index, class_name in enumerate(label_encoder.classes_)}
    print(f"  Encoded mapping: {label_mapping}")
    
    # 3. Train XGBoost Multiclass Classifier
    print("\n--- Phase 4: Training XGBClassifier on Semantic Features ---")
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=SEED,
        eval_metric="mlogloss",
        early_stopping_rounds=15,
        tree_method="hist",
        n_jobs=-1
    )
    
    t_train = time.time()
    clf.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=50
    )
    print(f"  Training finished in {time.time() - t_train:.2f}s. Best iteration: {clf.best_iteration}")
    
    # 4. Evaluate on test set
    print("\n--- Phase 5: Evaluating Model on Test Set ---")
    t_predict = time.time()
    y_pred = clf.predict(X_test)
    pred_labels = label_encoder.inverse_transform(y_pred)
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
        zero_division=0
    )
    
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )
    
    conf_matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(len(label_encoder.classes_))
    )
    
    print("\n" + "="*60)
    print("DETAILED MULTICLASS CLASSIFICATION REPORT (TEST SPLIT)")
    print("="*60)
    print(report)
    print("="*60)

    binary_metrics = save_binary_product_outputs(results_dir, test_labels, pred_labels)
    
    # 5. Save results
    print("Saving results and artifacts...")
    metrics = {
        "model": "CodeBERT + XGBoost",
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
        "device_used": str(device) if device is not None else "Cached",
        "binary_product_metrics_file": "binary_product_metrics.json",
        "binary_product_macro_f1": binary_metrics["macro_f1"],
        "binary_product_balanced_accuracy": binary_metrics["balanced_accuracy"],
        "binary_product_f1_review_required": binary_metrics["f1_review_required"],
        "execution_time_seconds": float(time.time() - start_time)
    }
    
    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    with open(results_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
        
    save_confusion_matrix(
        results_dir / "confusion_matrix.csv",
        conf_matrix,
        label_encoder.classes_
    )
    
    # Convert report to simple class metrics CSV
    with open(results_dir / "per_class_metrics.csv", "w", encoding="utf-8") as f:
        f.write("class,precision,recall,f1-score,support\n")
        for cls_name in label_encoder.classes_:
            val = report_dict[cls_name]
            f.write(f"{cls_name},{val['precision']:.4f},{val['recall']:.4f},{val['f1-score']:.4f},{val['support']}\n")
            
    # Save the model
    joblib.dump(
        {
            "model": clf,
            "label_encoder": label_encoder,
            "feature_count": int(X_train.shape[1])
        },
        results_dir / "model.joblib"
    )
    
    with open(results_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2)
        
    print(f"\nPipeline finished successfully in {time.time() - start_time:.2f}s.")

if __name__ == "__main__":
    main()
