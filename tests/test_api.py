import pytest
import numpy as np
import scipy.sparse as sp
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.model.predictor import SpamPredictor, PredictionResult
from src.data.preprocessor import TextPreprocessor


def make_mock_predictor(is_spam: bool = True, prob: float = 0.92) -> SpamPredictor:
    predictor = SpamPredictor.__new__(SpamPredictor)
    predictor.model_name = "MockModel"
    predictor.preprocessor = TextPreprocessor()
    predictor._feature_names = None
    predictor._spam_coefs = None

    model = MagicMock()
    label = 1 if is_spam else 0
    model.predict.return_value = np.array([label])
    spam_p = prob if is_spam else (1 - prob)
    model.predict_proba.return_value = np.array([[1 - spam_p, spam_p]])
    predictor.model = model

    vectorizer = MagicMock()
    vectorizer.transform.return_value = sp.csr_matrix(np.zeros((1, 10)))
    predictor.vectorizer = vectorizer

    return predictor


@pytest.fixture
def spam_client():
    app = create_app()
    app.state.predictor = make_mock_predictor(is_spam=True, prob=0.92)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def ham_client():
    app = create_app()
    app.state.predictor = make_mock_predictor(is_spam=False, prob=0.95)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def no_model_client():
    app = create_app()
    app.state.predictor = None
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    def test_health_ok(self, spam_client):
        resp = spam_client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    def test_health_degraded_when_no_model(self, no_model_client):
        resp = no_model_client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_loaded"] is False


class TestPredictEndpoint:
    def test_predict_spam(self, spam_client):
        resp = spam_client.post("/api/v1/predict", json={"message": "Win a free iPhone now!"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_spam"] is True
        assert data["label"] == "spam"
        assert "spam_probability" in data
        assert "top_spam_words" in data

    def test_predict_ham(self, ham_client):
        resp = ham_client.post("/api/v1/predict", json={"message": "Hey, dinner tonight?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_spam"] is False
        assert data["label"] == "ham"

    def test_predict_empty_message(self, spam_client):
        resp = spam_client.post("/api/v1/predict", json={"message": ""})
        assert resp.status_code == 422

    def test_predict_whitespace_only(self, spam_client):
        resp = spam_client.post("/api/v1/predict", json={"message": "   "})
        assert resp.status_code == 422

    def test_predict_missing_field(self, spam_client):
        resp = spam_client.post("/api/v1/predict", json={})
        assert resp.status_code == 422

    def test_predict_no_model(self, no_model_client):
        resp = no_model_client.post("/api/v1/predict", json={"message": "Test message"})
        assert resp.status_code == 503

    def test_predict_too_long_message(self, spam_client):
        resp = spam_client.post("/api/v1/predict", json={"message": "x" * 5001})
        assert resp.status_code == 422


class TestBatchPredictEndpoint:
    def test_batch_predict_returns_results(self, spam_client):
        import scipy.sparse as sp
        spam_client.app.state.predictor.model.predict.return_value = np.array([1, 0])
        spam_client.app.state.predictor.model.predict_proba.return_value = np.array(
            [[0.1, 0.9], [0.9, 0.1]]
        )
        spam_client.app.state.predictor.vectorizer.transform.return_value = sp.csr_matrix(
            np.zeros((2, 10))
        )
        resp = spam_client.post(
            "/api/v1/predict/batch",
            json={"messages": ["Free prize!", "Hello friend"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2
        assert "spam_count" in data
        assert "ham_count" in data

    def test_batch_predict_empty_list(self, spam_client):
        resp = spam_client.post("/api/v1/predict/batch", json={"messages": []})
        assert resp.status_code == 422

    def test_batch_predict_too_many(self, spam_client):
        messages = ["test message"] * 101
        resp = spam_client.post("/api/v1/predict/batch", json={"messages": messages})
        assert resp.status_code == 422
