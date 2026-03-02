import numpy as np
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from loguru import logger


@dataclass
class EvaluationResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: list
    classification_report: str

    def summary(self) -> str:
        return (
            f"[{self.model_name}] "
            f"Accuracy={self.accuracy:.4f} | "
            f"Precision={self.precision:.4f} | "
            f"Recall={self.recall:.4f} | "
            f"F1={self.f1:.4f} | "
            f"ROC-AUC={self.roc_auc:.4f}"
        )

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "roc_auc": round(self.roc_auc, 4),
            "confusion_matrix": self.confusion_matrix,
        }


def evaluate(model_name: str, y_true, y_pred, y_prob=None) -> EvaluationResult:
    """Compute and return all evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else float("nan")
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=["Ham", "Spam"])

    result = EvaluationResult(
        model_name=model_name,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        confusion_matrix=cm,
        classification_report=report,
    )

    logger.info(result.summary())
    logger.debug(f"\n{report}")

    return result
