import joblib
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

from src.data.preprocessor import TextPreprocessor


@dataclass
class PredictionResult:
    label: str                    # "spam" or "ham"
    is_spam: bool
    spam_probability: float
    ham_probability: float
    top_spam_words: list[str]
    top_ham_words: list[str]
    model_name: str

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "is_spam": self.is_spam,
            "spam_probability": round(self.spam_probability, 4),
            "ham_probability": round(self.ham_probability, 4),
            "top_spam_words": self.top_spam_words,
            "top_ham_words": self.top_ham_words,
            "model_name": self.model_name,
        }


class SpamPredictor:
    def __init__(self, model_dir: str = "models", preprocessor: Optional[TextPreprocessor] = None):
        self.model_dir = Path(model_dir)
        self.model = None
        self.vectorizer = None
        self.model_name = "unknown"
        self.preprocessor = preprocessor or TextPreprocessor()
        self._feature_names: Optional[np.ndarray] = None
        self._spam_coefs: Optional[np.ndarray] = None

    def load(self) -> "SpamPredictor":
        """Load model, vectorizer and metadata from disk."""
        model_path = self.model_dir / "model.joblib"
        vectorizer_path = self.model_dir / "vectorizer.joblib"
        meta_path = self.model_dir / "metadata.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}. Run 'make train' first."
            )

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

        if meta_path.exists():
            meta = joblib.load(meta_path)
            self.model_name = meta.get("model_name", "unknown")

        # Pre-compute feature names and coefficients for fast explainability
        self._feature_names = np.array(self.vectorizer.get_feature_names_out())
        if hasattr(self.model, "coef_"):
            # Works for LogisticRegression and LinearSVM
            self._spam_coefs = self.model.coef_[0]

        logger.info(f"Loaded model: {self.model_name} from {self.model_dir}")
        return self

    def _get_top_words(self, text_vector, top_n: int = 10) -> tuple[list[str], list[str]]:
        """Return top spam-contributing and ham-contributing words for this input."""
        if self._spam_coefs is None or self._feature_names is None:
            return [], []

        # Element-wise product: tfidf weight × coefficient
        dense = np.asarray(text_vector.todense()).flatten()
        scores = dense * self._spam_coefs

        nonzero = np.where(dense > 0)[0]
        if len(nonzero) == 0:
            return [], []

        nonzero_scores = scores[nonzero]
        nonzero_features = self._feature_names[nonzero]

        sorted_idx = np.argsort(nonzero_scores)

        top_spam = nonzero_features[sorted_idx[-top_n:][::-1]].tolist()
        top_ham = nonzero_features[sorted_idx[:top_n]].tolist()

        return top_spam, top_ham

    def predict(self, message: str, top_n: int = 10) -> PredictionResult:
        """Predict a single message."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        cleaned = self.preprocessor.clean(message)
        vector = self.vectorizer.transform([cleaned])
        prediction = int(self.model.predict(vector)[0])

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(vector)[0]
            spam_prob = float(proba[1])
            ham_prob = float(proba[0])
        else:
            # SGDClassifier (LinearSVM) — use decision function as a proxy
            decision = self.model.decision_function(vector)[0]
            spam_prob = float(1 / (1 + np.exp(-decision)))
            ham_prob = 1.0 - spam_prob

        top_spam_words, top_ham_words = self._get_top_words(vector, top_n)

        return PredictionResult(
            label="spam" if prediction == 1 else "ham",
            is_spam=bool(prediction == 1),
            spam_probability=spam_prob,
            ham_probability=ham_prob,
            top_spam_words=top_spam_words,
            top_ham_words=top_ham_words,
            model_name=self.model_name,
        )

    def predict_batch(self, messages: list[str], top_n: int = 10) -> list[PredictionResult]:
        """Predict multiple messages efficiently."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        cleaned = [self.preprocessor.clean(m) for m in messages]
        vectors = self.vectorizer.transform(cleaned)
        predictions = self.model.predict(vectors)

        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(vectors)
        else:
            decisions = self.model.decision_function(vectors)
            spam_probs = 1 / (1 + np.exp(-decisions))
            probas = np.column_stack([1 - spam_probs, spam_probs])

        results = []
        for i, (pred, proba) in enumerate(zip(predictions, probas)):
            top_spam, top_ham = self._get_top_words(vectors[i], top_n)
            results.append(
                PredictionResult(
                    label="spam" if pred == 1 else "ham",
                    is_spam=bool(pred == 1),
                    spam_probability=float(proba[1]),
                    ham_probability=float(proba[0]),
                    top_spam_words=top_spam,
                    top_ham_words=top_ham,
                    model_name=self.model_name,
                )
            )
        return results
