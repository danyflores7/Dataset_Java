#!/usr/bin/env python3
"""
CodeBERT state-of-the-art baseline for Java clone classification.

This pipeline fine-tunes microsoft/codebert-base on the existing train/valid/test
splits after dropping T4 from every model comparison. The primary task is
multiclass classification over:

    T0, T1, T2, VST3, ST3, MT3, WT3

It also exports a derived binary product-triage evaluation so CodeBERT can be
compared with product-oriented metrics without changing the primary model task.
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from clone_experiment_config import (
    EXCLUDED_CLASSES,
    MODEL_CLASSES,
    filter_model_records,
    to_binary_product_label,
)


SEED = 42
MODEL_NAME = "CodeBERT"
HF_MODEL_NAME = "microsoft/codebert-base"
BINARY_LABELS = ["no_alert", "review_required"]
CLASS_TO_ID = {class_name: index for index, class_name in enumerate(MODEL_CLASSES)}
ID_TO_CLASS = {index: class_name for class_name, index in CLASS_TO_ID.items()}
BINARY_TO_ID = {"no_alert": 0, "review_required": 1}
ID_TO_BINARY = {value: key for key, value in BINARY_TO_ID.items()}
PER_REVIEW_CLASS_ORDER = ["T1", "T2", "VST3", "ST3", "MT3", "WT3"]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Fine-tune CodeBERT as the SOTA baseline.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=script_dir / "model_ready_balanced",
        help="Directory containing train_balanced.jsonl, valid_balanced.jsonl, and test_balanced.jsonl.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=script_dir / "results" / "codebert_sota",
        help="Directory where CodeBERT outputs will be saved.",
    )
    parser.add_argument("--model-name", default=HF_MODEL_NAME, help="Hugging Face model id or local path.")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_jsonl(path: Path) -> list[dict]:
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


def maybe_limit(records: list[dict], limit: int | None) -> list[dict]:
    return records if limit is None else records[:limit]


def load_split(path: Path, split_name: str, limit: int | None = None) -> list[dict]:
    raw_records = load_jsonl(path)
    records = filter_model_records(raw_records, split_name)
    return maybe_limit(records, limit)


class ClonePairDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer: AutoTokenizer, max_length: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        encoded = self.tokenizer(
            record["func1"],
            record["func2"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(CLASS_TO_ID[record["clone_type"]], dtype=torch.long)
        return item


def class_weights(records: list[dict], device: torch.device) -> torch.Tensor:
    counts = np.zeros(len(MODEL_CLASSES), dtype=np.float32)
    for record in records:
        counts[CLASS_TO_ID[record["clone_type"]]] += 1.0
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def predict(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    probabilities = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predict"):
            batch = {key: value.to(device) for key, value in batch.items()}
            batch.pop("labels")
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1)
            predictions.extend(pred.cpu().numpy().tolist())
            probabilities.extend(probs.cpu().numpy().tolist())
    return np.array(predictions, dtype=np.int64), np.array(probabilities, dtype=np.float32)


def train(model, train_loader, valid_loader, train_records, valid_records, args, device) -> tuple[dict, int]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = max(len(train_loader) * args.epochs, 1)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights(train_records, device))

    best_state = None
    best_epoch = 0
    best_macro_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train epoch {epoch}"):
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")

            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())

        valid_pred, _ = predict(model, valid_loader, device)
        y_valid = [CLASS_TO_ID[record["clone_type"]] for record in valid_records]
        valid_macro_f1 = f1_score(y_valid, valid_pred, average="macro", zero_division=0)
        valid_accuracy = accuracy_score(y_valid, valid_pred)
        avg_loss = total_loss / max(len(train_loader), 1)
        print(
            f"Epoch {epoch}: train_loss={avg_loss:.4f} "
            f"valid_accuracy={valid_accuracy:.4f} valid_macro_f1={valid_macro_f1:.4f}"
        )

        if valid_macro_f1 > best_macro_f1:
            best_macro_f1 = valid_macro_f1
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_valid_macro_f1": best_macro_f1}, best_epoch


def flatten_multiclass_report(report_dict: dict) -> list[dict]:
    rows = []
    for class_name in MODEL_CLASSES:
        values = report_dict[class_name]
        rows.append(
            {
                "model": MODEL_NAME,
                "class": class_name,
                "precision": float(values["precision"]),
                "recall": float(values["recall"]),
                "f1_score": float(values["f1-score"]),
                "support": int(values["support"]),
            }
        )
    return rows


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def binary_metrics(y_true: list[str], y_pred: list[str]) -> dict:
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

    return {
        "accuracy": safe_divide(tn + tp, support),
        "precision_no_alert": precision_no_alert,
        "recall_no_alert": recall_no_alert,
        "f1_no_alert": f1_no_alert,
        "precision_review_required": precision_review,
        "recall_review_required": recall_review,
        "f1_review_required": f1_review,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": (recall_no_alert + recall_review) / 2,
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


def prediction_rows(records: list[dict], pred_ids: np.ndarray, probabilities: np.ndarray) -> list[dict]:
    rows = []
    for index, (record, pred_id, probs) in enumerate(zip(records, pred_ids, probabilities)):
        predicted_class = ID_TO_CLASS[int(pred_id)]
        rows.append(
            {
                "index": index,
                "clone_type": record["clone_type"],
                "predicted_multiclass": predicted_class,
                "correct_multiclass": record["clone_type"] == predicted_class,
                **{f"prob_{class_name}": float(probs[CLASS_TO_ID[class_name]]) for class_name in MODEL_CLASSES},
            }
        )
    return rows


def binary_prediction_rows(multiclass_rows: list[dict]) -> list[dict]:
    rows = []
    for row in multiclass_rows:
        true_binary = to_binary_product_label(row["clone_type"])
        pred_binary = to_binary_product_label(row["predicted_multiclass"])
        rows.append(
            {
                "index": row["index"],
                "clone_type": row["clone_type"],
                "predicted_multiclass": row["predicted_multiclass"],
                "binary_label": true_binary,
                "predicted_binary": pred_binary,
                "correct_binary": true_binary == pred_binary,
            }
        )
    return rows


def per_original_class_review_report(rows: list[dict]) -> list[dict]:
    report_rows = []
    for clone_type in PER_REVIEW_CLASS_ORDER:
        class_rows = [row for row in rows if row["clone_type"] == clone_type]
        support = len(class_rows)
        predicted_review = sum(row["predicted_binary"] == "review_required" for row in class_rows)
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


def save_confusion_matrix(path: Path, labels: list[str], matrix: list[list[int]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])


def main() -> None:
    start_time = time.time()
    args = parse_args()
    set_seed(SEED)

    data_dir = args.data_dir.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("STARTING CODEBERT SOTA MULTICLASS PIPELINE")
    print(f"Data source directory: {data_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Classes: {', '.join(MODEL_CLASSES)}")
    print(f"Excluded classes: {', '.join(sorted(EXCLUDED_CLASSES))}")

    train_records = load_split(data_dir / "train_balanced.jsonl", "Train", args.max_train_samples)
    valid_records = load_split(data_dir / "valid_balanced.jsonl", "Valid", args.max_valid_samples)
    test_records = load_split(data_dir / "test_balanced.jsonl", "Test", args.max_test_samples)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(MODEL_CLASSES),
        id2label=ID_TO_CLASS,
        label2id=CLASS_TO_ID,
        local_files_only=args.local_files_only,
    ).to(device)

    train_loader = DataLoader(
        ClonePairDataset(train_records, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        ClonePairDataset(valid_records, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        ClonePairDataset(test_records, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
    )

    train_summary, best_epoch = train(
        model,
        train_loader,
        valid_loader,
        train_records,
        valid_records,
        args,
        device,
    )

    print("Evaluating best checkpoint on test...")
    test_pred_ids, test_probabilities = predict(model, test_loader, device)
    y_test = [CLASS_TO_ID[record["clone_type"]] for record in test_records]
    test_accuracy = accuracy_score(y_test, test_pred_ids)
    test_macro_f1 = f1_score(y_test, test_pred_ids, average="macro", zero_division=0)
    test_weighted_f1 = f1_score(y_test, test_pred_ids, average="weighted", zero_division=0)

    report = classification_report(
        y_test,
        test_pred_ids,
        labels=list(range(len(MODEL_CLASSES))),
        target_names=MODEL_CLASSES,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_test,
        test_pred_ids,
        labels=list(range(len(MODEL_CLASSES))),
        target_names=MODEL_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    conf_matrix = confusion_matrix(
        y_test,
        test_pred_ids,
        labels=list(range(len(MODEL_CLASSES))),
    )

    multiclass_rows = prediction_rows(test_records, test_pred_ids, test_probabilities)
    binary_rows = binary_prediction_rows(multiclass_rows)
    y_binary_true = [row["binary_label"] for row in binary_rows]
    y_binary_pred = [row["predicted_binary"] for row in binary_rows]
    binary_metrics_values = binary_metrics(y_binary_true, y_binary_pred)
    per_review_rows = per_original_class_review_report(binary_rows)

    metrics = {
        "model": MODEL_NAME,
        "hf_model_name": args.model_name,
        "task": "multiclass_clone_classification",
        "classes": MODEL_CLASSES,
        "excluded_classes": sorted(EXCLUDED_CLASSES),
        "accuracy": float(test_accuracy),
        "macro_f1": float(test_macro_f1),
        "weighted_f1": float(test_weighted_f1),
        "train_size": len(train_records),
        "valid_size": len(valid_records),
        "test_size": len(test_records),
        "best_epoch": best_epoch,
        **train_summary,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "runtime_seconds": time.time() - start_time,
    }
    binary_metrics_payload = {
        "model": MODEL_NAME,
        "task": "binary_product_triage_derived_from_multiclass",
        "positive_class": "review_required",
        "negative_class": "no_alert",
        "excluded_classes": sorted(EXCLUDED_CLASSES),
        **binary_metrics_values,
        "test_size": len(test_records),
    }

    write_csv(results_dir / "predictions_test.csv", multiclass_rows)
    write_csv(results_dir / "per_class_metrics.csv", flatten_multiclass_report(report_dict))
    save_confusion_matrix(
        results_dir / "confusion_matrix.csv",
        MODEL_CLASSES,
        conf_matrix.tolist(),
    )
    with open(results_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    save_json(results_dir / "metrics.json", metrics)

    write_csv(results_dir / "predictions_test_binary.csv", binary_rows)
    write_csv(results_dir / "per_original_class_review_recall_test.csv", per_review_rows)
    save_confusion_matrix(
        results_dir / "binary_confusion_matrix.csv",
        BINARY_LABELS,
        binary_metrics_values["confusion_matrix"]["values"],
    )
    save_json(results_dir / "binary_product_metrics.json", binary_metrics_payload)

    with open(results_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({str(index): class_name for index, class_name in ID_TO_CLASS.items()}, f, indent=2)

    model.save_pretrained(results_dir / "model")
    tokenizer.save_pretrained(results_dir / "model")

    print("Multiclass test metrics:")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Macro F1: {test_macro_f1:.4f}")
    print(f"  Weighted F1: {test_weighted_f1:.4f}")
    print("Derived binary product metrics:")
    print(f"  Macro F1: {binary_metrics_payload['macro_f1']:.4f}")
    print(f"  Balanced accuracy: {binary_metrics_payload['balanced_accuracy']:.4f}")
    print(f"  Recall no_alert: {binary_metrics_payload['recall_no_alert']:.4f}")
    print(f"  Recall review_required: {binary_metrics_payload['recall_review_required']:.4f}")
    print(f"  False positive rate: {binary_metrics_payload['false_positive_rate']:.4f}")
    print(f"Saved results to: {results_dir}")


if __name__ == "__main__":
    main()
