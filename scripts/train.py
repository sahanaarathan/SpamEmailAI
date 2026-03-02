"""
Training script — run from project root:

    python scripts/train.py
    python scripts/train.py --config configs/config.yaml
"""

import argparse
import sys
import os
import yaml
from pathlib import Path
from loguru import logger

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.trainer import ModelTrainer


def setup_logging(config: dict):
    log_cfg = config.get("logging", {})
    log_dir = log_cfg.get("dir", "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level=log_cfg.get("level", "INFO"), colorize=True)
    logger.add(
        f"{log_dir}/train.log",
        level=log_cfg.get("level", "INFO"),
        rotation=log_cfg.get("rotation", "10 MB"),
        retention=log_cfg.get("retention", "1 week"),
    )


def main():
    parser = argparse.ArgumentParser(description="Train the Spam Email AI model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML (default: configs/config.yaml)",
    )
    args = parser.parse_args()

    config_path = ROOT / args.config
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    setup_logging(config)
    logger.info(f"Using config: {config_path}")

    trainer = ModelTrainer(config)
    model, vectorizer, best_result = trainer.run()

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(best_result.summary())
    logger.info("=" * 60)
    logger.info(f"\n{best_result.classification_report}")

    # Print all model comparisons
    logger.info("All model results:")
    for result in trainer.evaluation_results:
        logger.info(f"  {result.summary()}")


if __name__ == "__main__":
    main()
