import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.model.predictor import SpamPredictor, PredictionResult
from src.data.preprocessor import TextPreprocessor


def make_mock_predictor() -> SpamPredictor:
    """Build a SpamPredictor with mocked internals."""
    predictor = SpamPredictor.__new__(SpamPredictor)
    predictor.model_dir = MagicMock()
    predictor.model_name = "MockModel"
    predictor.preprocessor = TextPreprocessor()
    predictor._feature_names = None
    predictor._spam_coefs = None

    # Mock model
    model = MagicMock()
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.1, 0.9]])
    predictor.model = model

    # Mock vectorizer
    vectorizer = MagicMock()
    import scipy.sparse as sp
    vectorizer.transform.return_value = sp.csr_matrix(np.zeros((1, 10)))
    predictor.vectorizer = vectorizer

    return predictor


class TestPredictionResult:
    def test_to_dict_keys(self):
        result = PredictionResult(
            label="spam",
            is_spam=True,
            spam_probability=0.95,
            ham_probability=0.05,
            top_spam_words=["free", "win"],
            top_ham_words=["hello"],
            model_name="TestModel",
        )
        d = result.to_dict()
        assert "label" in d
        assert "is_spam" in d
        assert "spam_probability" in d
        assert "ham_probability" in d
        assert "top_spam_words" in d
        assert "top_ham_words" in d
        assert "model_name" in d

    def test_probabilities_sum_to_one(self):
        result = PredictionResult(
            label="ham",
            is_spam=False,
            spam_probability=0.03,
            ham_probability=0.97,
            top_spam_words=[],
            top_ham_words=[],
            model_name="TestModel",
        )
        assert abs(result.spam_probability + result.ham_probability - 1.0) < 0.01


class TestSpamPredictor:
    def test_predict_returns_prediction_result(self):
        predictor = make_mock_predictor()
        result = predictor.predict("Free prize click now")
        assert isinstance(result, PredictionResult)
        assert result.label in ("spam", "ham")

    def test_predict_spam(self):
        predictor = make_mock_predictor()
        predictor.model.predict.return_value = np.array([1])
        predictor.model.predict_proba.return_value = np.array([[0.05, 0.95]])
        result = predictor.predict("Win a free iPhone")
        assert result.is_spam is True
        assert result.label == "spam"

    def test_predict_ham(self):
        predictor = make_mock_predictor()
        predictor.model.predict.return_value = np.array([0])
        predictor.model.predict_proba.return_value = np.array([[0.95, 0.05]])
        result = predictor.predict("Hey, dinner tonight?")
        assert result.is_spam is False
        assert result.label == "ham"

    def test_predict_raises_without_model(self):
        predictor = SpamPredictor.__new__(SpamPredictor)
        predictor.model = None
        predictor.vectorizer = None
        predictor.preprocessor = TextPreprocessor()
        predictor._feature_names = None
        predictor._spam_coefs = None
        with pytest.raises(RuntimeError, match="Model not loaded"):
            predictor.predict("test message")

    def test_predict_batch_returns_list(self):
        predictor = make_mock_predictor()
        import scipy.sparse as sp
        predictor.model.predict.return_value = np.array([1, 0])
        predictor.model.predict_proba.return_value = np.array([[0.1, 0.9], [0.9, 0.1]])
        predictor.vectorizer.transform.return_value = sp.csr_matrix(np.zeros((2, 10)))
        results = predictor.predict_batch(["spam message", "hello there"])
        assert len(results) == 2
        assert all(isinstance(r, PredictionResult) for r in results)

    def test_predict_batch_empty_raises_or_handles(self):
        predictor = make_mock_predictor()
        import scipy.sparse as sp
        predictor.model.predict.return_value = np.array([])
        predictor.model.predict_proba.return_value = np.array([]).reshape(0, 2)
        predictor.vectorizer.transform.return_value = sp.csr_matrix(np.zeros((0, 10)))
        results = predictor.predict_batch([])
        assert results == []

    def test_load_raises_file_not_found(self, tmp_path):
        predictor = SpamPredictor(model_dir=str(tmp_path / "nonexistent"))
        with pytest.raises(FileNotFoundError):
            predictor.load()
