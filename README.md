# Spam Email AI

A production-grade spam detection system built with Machine Learning, FastAPI, and Streamlit.

## Features

- **Multi-model training** — compares Logistic Regression, Linear SVM, and Naive Bayes; selects the best by F1-score
- **Word-level explainability** — shows which words contributed most to spam/ham classification
- **REST API** — FastAPI backend with single + batch prediction, rate limiting, Swagger UI
- **Streamlit UI** — rich interface with probability gauge, word highlights, batch upload, history
- **Fully tested** — pytest suite with 70%+ coverage
- **Docker-ready** — multi-stage Dockerfile, docker-compose for API + UI
- **CI/CD** — GitHub Actions: lint → test → build → deploy

## Project Structure

```
SpamEmailAI/
├── src/
│   ├── data/
│   │   ├── loader.py          # Dataset loading & validation
│   │   └── preprocessor.py    # Text cleaning pipeline
│   ├── model/
│   │   ├── trainer.py         # Multi-model training & comparison
│   │   ├── evaluator.py       # Metrics (accuracy, F1, ROC-AUC)
│   │   └── predictor.py       # Inference + word explainability
│   ├── api/
│   │   ├── main.py            # FastAPI app factory
│   │   ├── routes/predict.py  # /predict, /predict/batch, /health
│   │   └── schemas/request.py # Pydantic request/response models
│   └── ui/
│       └── app.py             # Streamlit web interface
├── tests/
│   ├── test_preprocessor.py
│   ├── test_predictor.py
│   └── test_api.py
├── scripts/
│   ├── train.py               # Training script
│   └── evaluate.py            # Evaluation script
├── configs/
│   └── config.yaml            # All hyperparameters & settings
├── models/                    # Saved model artifacts (gitignored)
├── logs/                      # Log files (gitignored)
├── Dockerfile                 # API image
├── Dockerfile.ui              # UI image
├── docker-compose.yml
└── Makefile                   # Developer commands
```

## Quick Start

### 1. Install dependencies

```bash
make install-dev
```

### 2. Train the model

```bash
make train
```

This trains Logistic Regression, Linear SVM, and Naive Bayes, picks the best by F1-score, and saves it to `models/`.

### 3. Run the app

**Streamlit UI (direct mode — no API needed):**
```bash
make run-ui
```

**FastAPI backend:**
```bash
make run-api
# Docs at http://localhost:8000/docs
```

**Streamlit UI in API mode:**
```bash
make run-ui-api
```

## Docker

```bash
# Train first (mounts models/ from host)
make docker-train

# Start API + UI
make docker-up
# API:  http://localhost:8000/docs
# UI:   http://localhost:8501
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/predict` | Classify a single message |
| POST | `/api/v1/predict/batch` | Classify up to 100 messages |

**Single prediction example:**
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Congratulations! You have won a free prize. Click now."}'
```

**Response:**
```json
{
  "label": "spam",
  "is_spam": true,
  "spam_probability": 0.9842,
  "ham_probability": 0.0158,
  "top_spam_words": ["congratulations", "won", "free", "prize", "click"],
  "top_ham_words": [],
  "model_name": "LogisticRegression"
}
```

## Development

```bash
make test        # Run tests
make test-cov    # Tests + HTML coverage report
make lint        # Check code style
make format      # Auto-format code
make evaluate    # Evaluate saved model on test set
make clean       # Remove generated files
```

## Configuration

All settings are in `configs/config.yaml`:

- `data` — dataset path, train/test split ratio
- `preprocessing` — short-form expansions, max input length
- `model` — TF-IDF params, per-model hyperparameters
- `api` — host, port, rate limits
- `logging` — level, rotation, retention

## CI/CD

GitHub Actions pipeline (`.github/workflows/ci.yml`):
1. **Lint** — black, isort, flake8
2. **Test** — pytest with coverage
3. **Docker build** — validates both images build successfully
4. **Deploy** — pushes to Docker Hub on merge to `main` (requires secrets)

Set `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` in GitHub repository secrets to enable deployment.
