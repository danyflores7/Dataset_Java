#!/usr/bin/env python3
"""
Qualitative Error Analysis: False Positives & False Negatives of the Hybrid Model.

This script loads the hybrid TF-IDF + AST + XGBoost model, runs inference on the
test split, and outputs details for:
1. False Positives (Actual: T0 | Predicted: Clone)
2. False Negatives (Actual: Clone | Predicted: T0)
"""

import os
import sys
import json
import joblib
import importlib
import numpy as np
from pathlib import Path

def main():
    script_dir = Path(__file__).resolve().parent
    sys.path.append(str(script_dir))
    
    # Dynamic import for module with '+' in the name
    try:
        hybrid_module = importlib.import_module("AST+IDFhybrid_xgboost_pipeline")
        build_tfidf_pairwise_features = hybrid_module.build_tfidf_pairwise_features
    except ModuleNotFoundError:
        print("Error: Could not import AST+IDFhybrid_xgboost_pipeline. Please ensure it is in the same directory.")
        return

    from ast_xgboost_pipeline import load_jsonl, extract_split_features

    results_dir = script_dir / "results" / "hybrid_xgboost"
    model_path = results_dir / "model.joblib"
    
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}. Please train the hybrid model first.")
        return
        
    print("Loading hybrid model...")
    model_data = joblib.load(model_path)
    clf = model_data["model"]
    vectorizer = model_data["tfidf_vectorizer"]
    label_encoder = model_data["label_encoder"]
    
    test_path = script_dir / "model_ready_balanced" / "test_balanced.jsonl"
    if not test_path.exists():
        print(f"Error: Test data not found at {test_path}.")
        return
        
    print("Loading test dataset...")
    test_data = load_jsonl(str(test_path))
    
    print("Extracting TF-IDF features...")
    test_f1 = [record["func1"] for record in test_data]
    test_f2 = [record["func2"] for record in test_data]
    test_labels = [record["clone_type"] for record in test_data]
    
    X_test_tfidf = build_tfidf_pairwise_features(
        vectorizer.transform(test_f1),
        vectorizer.transform(test_f2)
    )
    
    print("Extracting AST features...")
    X_test_ast, _ = extract_split_features(test_data, "Test")
    
    # Combine
    X_test = np.hstack([X_test_tfidf, X_test_ast]).astype(np.float32)
    
    print("Predicting...")
    y_test = label_encoder.transform(test_labels)
    y_pred = clf.predict(X_test)
    
    # 1. False Positives: Actual T0 but predicted as something else
    t0_encoded = label_encoder.transform(["T0"])[0]
    false_pos_idx = np.where((y_test == t0_encoded) & (y_pred != t0_encoded))[0]
    
    # 2. False Negatives: Actual Clone (any) but predicted as T0
    false_neg_idx = np.where((y_test != t0_encoded) & (y_pred == t0_encoded))[0]
    
    print("\n" + "=" * 60)
    print("ANALYSIS OF T0 FALSE POSITIVES (Actual: T0 | Predicted: Clone)")
    print("=" * 60)
    print(f"Found {len(false_pos_idx)} False Positives out of {test_labels.count('T0')} actual T0 pairs.\n")
    
    for count, idx in enumerate(false_pos_idx, start=1):
        pred_label = label_encoder.inverse_transform([y_pred[idx]])[0]
        record = test_data[idx]
        cosine_sim = X_test_tfidf[idx, -1]
        
        print(f"FP #{count} - Index: {idx} | Predicted Class: {pred_label} | Cosine Similarity (TF-IDF): {cosine_sim:.4f}")
        print("-" * 50)
        print(f"Function 1 (code excerpt):\n{record['func1'][:250]}...")
        print("-" * 50)
        print(f"Function 2 (code excerpt):\n{record['func2'][:250]}...")
        print("=" * 60)

    print("\n" + "=" * 60)
    print("ANALYSIS OF T0 FALSE NEGATIVES (Actual: Clone | Predicted: T0)")
    print("=" * 60)
    print(f"Found {len(false_neg_idx)} False Negatives.\n")
    
    for count, idx in enumerate(false_neg_idx, start=1):
        actual_label = test_labels[idx]
        record = test_data[idx]
        cosine_sim = X_test_tfidf[idx, -1]
        
        print(f"FN #{count} - Index: {idx} | Actual Class: {actual_label} | Cosine Similarity (TF-IDF): {cosine_sim:.4f}")
        print("-" * 50)
        print(f"Function 1 (code excerpt):\n{record['func1'][:250]}...")
        print("-" * 50)
        print(f"Function 2 (code excerpt):\n{record['func2'][:250]}...")
        print("=" * 60)

if __name__ == "__main__":
    main()
