import pandas as pd
from loguru import logger
from pathlib import Path


def load_dataset(path: str, encoding: str = "latin-1") -> pd.DataFrame:
    """Load and validate the spam dataset."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {file_path.resolve()}")

    logger.info(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path, encoding=encoding)

    # Keep only relevant columns
    df = df[["v1", "v2"]].copy()
    df.columns = ["label", "message"]

    # Drop nulls and duplicates
    original_len = len(df)
    df.dropna(subset=["label", "message"], inplace=True)
    df.drop_duplicates(subset=["message"], inplace=True)
    logger.info(f"Removed {original_len - len(df)} null/duplicate rows")

    # Validate labels
    valid_labels = {"ham", "spam"}
    invalid = set(df["label"].unique()) - valid_labels
    if invalid:
        raise ValueError(f"Unexpected labels in dataset: {invalid}")

    # Encode labels
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    spam_count = df["label"].sum()
    ham_count = len(df) - spam_count
    logger.info(
        f"Dataset loaded: {len(df)} rows | Ham: {ham_count} | Spam: {spam_count}"
    )

    return df
