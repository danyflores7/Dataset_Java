"""Shared experiment configuration for Java clone model pipelines."""

EXCLUDED_CLASSES = {"T4"}
MODEL_CLASSES = ["T0", "T1", "T2", "VST3", "ST3", "MT3", "WT3"]
REVIEW_REQUIRED_CLASSES = {"T1", "T2", "VST3", "ST3", "MT3", "WT3"}
NO_ALERT_CLASS = "T0"


def is_model_class(clone_type: str) -> bool:
    return clone_type in MODEL_CLASSES


def filter_model_records(records: list[dict], split_name: str = "") -> list[dict]:
    """Drop classes excluded from all model training/evaluation."""
    filtered = [record for record in records if is_model_class(record["clone_type"])]
    removed = len(records) - len(filtered)
    if split_name:
        print(f"  {split_name}: excluded {removed:,} T4 rows; kept {len(filtered):,} rows")
    return filtered


def to_binary_product_label(clone_type: str) -> str:
    if clone_type == NO_ALERT_CLASS:
        return "no_alert"
    if clone_type in REVIEW_REQUIRED_CLASSES:
        return "review_required"
    raise ValueError(f"Class {clone_type!r} is excluded from model evaluation.")
