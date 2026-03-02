"""
Evaluate a saved model on test data.

    python scripts/evaluate.py
    python scripts/evaluate.py --config configs/config.yaml
"""

import argparse
import sys
import yaml
import json
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_dataset
from src.data.preprocessor import TextPreprocessor
from src.model.predictor import SpamPredictor
from src.model.evaluator import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate the saved Spam Email AI model.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    config_path = ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True)

    # Load data
    df = load_dataset(config["data"]["path"], config["data"].get("encoding", "latin-1"))

    preprocessor = TextPreprocessor(
        short_forms=config["preprocessing"].get("short_forms"),
        max_input_length=config["preprocessing"].get("max_input_length", 5000),
    )
    df["message"] = preprocessor.clean_series(df["message"])

    _, X_test_text, _, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=config["data"].get("test_size", 0.2),
        random_state=config["data"].get("random_state", 42),
        stratify=df["label"],
    )

    # Load saved model
    predictor = SpamPredictor(
        model_dir=config["model"]["save_dir"],
        preprocessor=preprocessor,
    ).load()

    # Vectorize and predict
    X_test = predictor.vectorizer.transform(X_test_text)
    y_pred = predictor.model.predict(X_test)
    y_prob = None
    if hasattr(predictor.model, "predict_proba"):
        y_prob = predictor.model.predict_proba(X_test)[:, 1]

    result = evaluate(predictor.model_name, y_test, y_pred, y_prob)

    logger.info("=" * 60)
    logger.info(result.summary())
    logger.info("=" * 60)
    logger.info(f"\n{result.classification_report}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
