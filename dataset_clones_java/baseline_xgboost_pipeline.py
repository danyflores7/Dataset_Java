#!/usr/bin/env python3
"""
Baseline XGBoost Pipeline for Java Code Clone Multiclass Classification

This script implements a complete end-to-end baseline pipeline:
1. Sets seed for reproducibility.
2. Performs a memory-efficient 1st pass to scan class distribution & record line indices.
3. Randomly undersamples majority classes (T0, WT3, MT3) to handle extreme class imbalance.
4. Performs a 2nd pass loading the actual code text only for the selected subset.
5. Splits the balanced dataset into stratified Train/Valid/Test partitions.
6. Fits a TF-IDF vectorizer on the training code and computes pairwise symmetric features:
   - Absolute difference: |v1 - v2|
   - Element-wise product: v1 * v2
   - Cosine similarity: (v1 . v2) / (||v1|| * ||v2||)
7. Trains a multiclass XGBoost classifier (with early stopping on validation).
8. Evaluates the classifier and prints classification reports for all clone types.
"""

import os
import json
import random
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# 1. CONFIGURATION & REPRODUCIBILITY SEED
# ==========================================
SEED = 42
MAX_SAMPLES_PER_CLASS = 3000  # Cap to balance dataset and run in minutes
TFIDF_MAX_FEATURES = 1000     # Vocabulary size for TF-IDF

# Set global random state for reproducibility
random.seed(SEED)
np.random.seed(SEED)

def extract_clone_type(line):
    """
    Highly optimized string-based parser to extract clone_type from JSONL lines
    without parsing the entire JSON object (saving substantial memory & time).
    """
    idx = line.rfind('"clone_type":')
    if idx == -1:
        return "UNKNOWN"
    start_quote = line.find('"', idx + 13)
    if start_quote == -1:
        return "UNKNOWN"
    end_quote = line.find('"', start_quote + 1)
    if end_quote == -1:
        return "UNKNOWN"
    return line[start_quote+1:end_quote]

def build_pairwise_features(X1_sparse, X2_sparse):
    """
    Constructs a symmetric feature representation of code fragment pairs:
    [ |v1 - v2|, v1 * v2, cosine_similarity(v1, v2) ]
    """
    t0 = time.time()
    # Convert sparse matrices to dense arrays for vectorized numpy operations
    v1 = X1_sparse.toarray()
    v2 = X2_sparse.toarray()
    
    # 1. Absolute Difference (symmetric)
    abs_diff = np.abs(v1 - v2)
    
    # 2. Element-wise Product (symmetric)
    product = v1 * v2
    
    # 3. Cosine Similarity (symmetric)
    dot_product = np.sum(v1 * v2, axis=1)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    epsilon = 1e-9  # Avoid division by zero
    cosine_sim = dot_product / (norm1 * norm2 + epsilon)
    cosine_sim = cosine_sim.reshape(-1, 1)
    
    # Concatenate features horizontally
    features = np.hstack([abs_diff, product, cosine_sim])
    print(f"  Feature shape: {features.shape} generated in {time.time() - t0:.2f}s")
    return features

def main():
    start_time = time.time()
    
    # Resolve directory paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_ready_dir = os.path.join(script_dir, "model_ready")
    
    print("=============================================================")
    print("STARTING BASELINE XGBOOST PIPELINE FOR CODE CLONE DETECTION")
    print("=============================================================")
    print(f"Data source directory: {model_ready_dir}")
    print(f"Reproducibility Seed: {SEED}")
    print(f"Max samples per class (Undersampling cap): {MAX_SAMPLES_PER_CLASS}")
    
    jsonl_files = ["train.jsonl", "valid.jsonl", "test.jsonl"]
    
    # ==========================================
    # 2. FIRST PASS: SCAN CLASS DISTRIBUTION
    # ==========================================
    print("\n--- Phase 1: Scanning Dataset Class Distribution (Memory-Efficient Pass) ---")
    class_to_lines = defaultdict(list)
    
    for fname in jsonl_files:
        fpath = os.path.join(model_ready_dir, fname)
        if not os.path.exists(fpath):
            print(f"Error: {fpath} does not exist. Please check your data paths.")
            return
        
        print(f"Scanning {fname}...")
        t_scan = time.time()
        with open(fpath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                ctype = extract_clone_type(line)
                # Keep track of file and line index
                class_to_lines[ctype].append((fname, idx))
        print(f"  Scanned in {time.time() - t_scan:.2f}s.")
        
    print("\nOriginal Class Distribution:")
    for ctype, items in sorted(class_to_lines.items()):
        print(f"  {ctype}: {len(items):,}")
        
    # ==========================================
    # 3. UNDERSAMPLING MAJORITY CLASSES
    # ==========================================
    print("\n--- Phase 2: Applying Random Undersampling to Majority Classes ---")
    selected_indices = []
    
    for ctype, items in sorted(class_to_lines.items()):
        if len(items) > MAX_SAMPLES_PER_CLASS:
            chosen = random.sample(items, MAX_SAMPLES_PER_CLASS)
            print(f"  Class {ctype}: Undersampled {len(items):,} -> {MAX_SAMPLES_PER_CLASS}")
        else:
            chosen = items
            print(f"  Class {ctype}: Kept all {len(items):,} samples")
        selected_indices.extend(chosen)
        
    print(f"Total selected samples after balancing: {len(selected_indices):,}")
    
    # Group indices by file to read each file sequentially
    selected_by_file = defaultdict(list)
    for fname, idx in selected_indices:
        selected_by_file[fname].append(idx)
        
    # Sort indices within each file to allow single linear scan
    for fname in selected_by_file:
        selected_by_file[fname].sort()
        
    # ==========================================
    # 4. SECOND PASS: LOAD SUBSET CODE TEXT
    # ==========================================
    print("\n--- Phase 3: Loading Code Text for Selected Balanced Subset ---")
    loaded_data = []
    
    for fname, indices in selected_by_file.items():
        fpath = os.path.join(model_ready_dir, fname)
        indices_set = set(indices)
        loaded_count = 0
        t_load = time.time()
        
        with open(fpath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx in indices_set:
                    obj = json.loads(line)
                    loaded_data.append({
                        "func1": obj["func1"],
                        "func2": obj["func2"],
                        "clone_type": obj["clone_type"]
                    })
                    loaded_count += 1
        print(f"  Loaded {loaded_count:,} samples from {fname} in {time.time() - t_load:.2f}s")
        
    # ==========================================
    # 5. STRATIFIED DIVISIÓN OF DATA (SPLITS)
    # ==========================================
    print("\n--- Phase 4: Splitting Balanced Dataset (Train: 70%, Valid: 15%, Test: 15%) ---")
    
    # 1st split: Train and Temp (30%)
    train_data, temp_data = train_test_split(
        loaded_data,
        test_size=0.30,
        random_state=SEED,
        stratify=[x["clone_type"] for x in loaded_data]
    )
    
    # 2nd split: Valid (15%) and Test (15%)
    valid_data, test_data = train_test_split(
        temp_data,
        test_size=0.50,
        random_state=SEED,
        stratify=[x["clone_type"] for x in temp_data]
    )
    
    print(f"  Train size: {len(train_data):,}")
    print(f"  Validation size: {len(valid_data):,}")
    print(f"  Test size: {len(test_data):,}")
    
    # Verify split distributions
    def print_distribution(split_name, data_list):
        counts = defaultdict(int)
        for x in data_list:
            counts[x["clone_type"]] += 1
        dist_str = ", ".join([f"{k}: {v}" for k, v in sorted(counts.items())])
        print(f"    {split_name} distribution: {dist_str}")
        
    print_distribution("Train", train_data)
    print_distribution("Valid", valid_data)
    print_distribution("Test", test_data)
    
    # ==========================================
    # 4.5. SAVE BALANCED SUBSET TO DISK
    # ==========================================
    print("\n--- Phase 4.5: Saving Balanced Splits to Disk (for GitHub / Team sharing) ---")
    balanced_out_dir = os.path.join(script_dir, "model_ready_balanced")
    os.makedirs(balanced_out_dir, exist_ok=True)
    
    balanced_files = {
        "train_balanced.jsonl": train_data,
        "valid_balanced.jsonl": valid_data,
        "test_balanced.jsonl": test_data
    }
    
    for filename, data_list in balanced_files.items():
        out_filepath = os.path.join(balanced_out_dir, filename)
        with open(out_filepath, "w", encoding="utf-8") as f:
            for item in data_list:
                f.write(json.dumps(item) + "\n")
        print(f"  Saved {len(data_list):,} samples to {out_filepath}")

    # Extract code text and target labels for each split
    train_f1 = [x["func1"] for x in train_data]
    train_f2 = [x["func2"] for x in train_data]
    train_labels = [x["clone_type"] for x in train_data]
    
    valid_f1 = [x["func1"] for x in valid_data]
    valid_f2 = [x["func2"] for x in valid_data]
    valid_labels = [x["clone_type"] for x in valid_data]
    
    test_f1 = [x["func1"] for x in test_data]
    test_f2 = [x["func2"] for x in test_data]
    test_labels = [x["clone_type"] for x in test_data]
    
    # ==========================================
    # 6. FEATURE EXTRACTION (TF-IDF & SYMMETRIC PAIRS)
    # ==========================================
    print("\n--- Phase 5: TF-IDF Feature Extraction & Symmetric Combination ---")
    print(f"  Fitting TfidfVectorizer (max_features={TFIDF_MAX_FEATURES})...")
    
    # Fit the vectorizer on the combined corpus of training functions
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        token_pattern=r'(?u)\b\w+\b'  # Standard word boundaries
    )
    # Combine training functions for fitting
    train_corpus = train_f1 + train_f2
    vectorizer.fit(train_corpus)
    
    print("  Transforming training splits...")
    X1_train = vectorizer.transform(train_f1)
    X2_train = vectorizer.transform(train_f2)
    
    print("  Transforming validation splits...")
    X1_valid = vectorizer.transform(valid_f1)
    X2_valid = vectorizer.transform(valid_f2)
    
    print("  Transforming test splits...")
    X1_test = vectorizer.transform(test_f1)
    X2_test = vectorizer.transform(test_f2)
    
    print("  Generating paired features (absolute diff, element product, cosine)...")
    print("    Train features:")
    X_train = build_pairwise_features(X1_train, X2_train)
    print("    Validation features:")
    X_valid = build_pairwise_features(X1_valid, X2_valid)
    print("    Test features:")
    X_test = build_pairwise_features(X1_test, X2_test)
    
    # ==========================================
    # 7. LABEL ENCODING
    # ==========================================
    print("\n--- Phase 6: Encoding Multi-class Target Labels ---")
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_valid = label_encoder.transform(valid_labels)
    y_test = label_encoder.transform(test_labels)
    
    classes_mapping = {idx: name for idx, name in enumerate(label_encoder.classes_)}
    print(f"  Encoded mapping: {classes_mapping}")
    
    # ==========================================
    # 8. TRAINING XGBOOST CLASSIFIER
    # ==========================================
    print("\n--- Phase 7: Training Multiclass XGBClassifier ---")
    
    # Configure multiclass XGBoost classifier
    # Multiclass classification uses multi:softprob internally
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=SEED,
        eval_metric="mlogloss",
        early_stopping_rounds=15,
        tree_method="hist",  # Fast histogram method
        n_jobs=-1            # Use all available CPU cores
    )
    
    t_train = time.time()
    clf.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=50  # Print progress every 50 rounds
    )
    print(f"  Training finished in {time.time() - t_train:.2f}s. Best iteration: {clf.best_iteration}")
    
    # ==========================================
    # 9. EVALUATION & CLASSIFICATION REPORT
    # ==========================================
    print("\n--- Phase 8: Evaluating Model Performance on Test Set ---")
    
    t_predict = time.time()
    y_pred = clf.predict(X_test)
    predict_time = time.time() - t_predict
    
    acc = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {acc:.4f} (Inference time: {predict_time:.3f}s)")
    
    print("\n" + "="*60)
    print("DETAILED MULTICLASS CLASSIFICATION REPORT (TEST SPLIT)")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, digits=4))
    print("="*60)
    
    print(f"\nPipeline finished successfully in {time.time() - start_time:.2f}s.")

if __name__ == "__main__":
    main()
