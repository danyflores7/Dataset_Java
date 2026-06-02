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
import time
import random
import numpy as np
import xgboost as xgb
import torch
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Ensure we can import from local path
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir))

from ast_xgboost_pipeline import load_jsonl, SEED, save_confusion_matrix

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Configurations
MODEL_NAME = "microsoft/codebert-base"
BATCH_SIZE = 32
MAX_LENGTH = 512  # CodeBERT context length limit

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

def get_or_load_embeddings(split_name, records, data_dir, cache_dir, tokenizer, model, device):
    """Load cached embeddings from disk if available, otherwise calculate and cache them."""
    emb1_path = cache_dir / f"{split_name}_emb1.npy"
    emb2_path = cache_dir / f"{split_name}_emb2.npy"
    
    if emb1_path.exists() and emb2_path.exists():
        print(f"  Found cached embeddings for {split_name} split. Loading...")
        emb1 = np.load(emb1_path)
        emb2 = np.load(emb2_path)
        return emb1, emb2
        
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

def main():
    start_time = time.time()
    data_dir = script_dir / "model_ready_balanced"
    results_dir = script_dir / "results" / "codebert_xgboost"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Cache directory for numpy embeddings
    cache_dir = results_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("=============================================================")
    print("STARTING CODEBERT + XGBOOST SEMANTIC CLASSIFICATION PIPELINE")
    print("=============================================================")
    print(f"Data source directory: {data_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Reproducibility Seed: {SEED}")
    
    # 1. Load balanced datasets
    print("Loading balanced splits...")
    train_data = load_jsonl(str(data_dir / "train_balanced.jsonl"))
    valid_data = load_jsonl(str(data_dir / "valid_balanced.jsonl"))
    test_data = load_jsonl(str(data_dir / "test_balanced.jsonl"))
    
    print(f"  Train size: {len(train_data):,}")
    print(f"  Valid size: {len(valid_data):,}")
    print(f"  Test size: {len(test_data):,}")
    
    # Check if cache exists for all splits to avoid loading Hugging Face model
    cache_exists = all(
        (cache_dir / f"{split}_emb1.npy").exists() and (cache_dir / f"{split}_emb2.npy").exists()
        for split in ["train", "valid", "test"]
    )
    
    device = None
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
        
    print("\n--- Phase 1: Extracting Semantic Embeddings ---")
    train_emb1, train_emb2 = get_or_load_embeddings("train", train_data, data_dir, cache_dir, tokenizer, model, device)
    valid_emb1, valid_emb2 = get_or_load_embeddings("valid", valid_data, data_dir, cache_dir, tokenizer, model, device)
    test_emb1, test_emb2 = get_or_load_embeddings("test", test_data, data_dir, cache_dir, tokenizer, model, device)
    
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
    
    # 5. Save results
    print("Saving results and artifacts...")
    metrics = {
        "model": "CodeBERT + XGBoost",
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "train_size": len(train_data),
        "valid_size": len(valid_data),
        "test_size": len(test_data),
        "feature_count": int(X_train.shape[1]),
        "device_used": str(device) if device is not None else "Cached",
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
