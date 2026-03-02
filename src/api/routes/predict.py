from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from src.api.schemas.request import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
)

router = APIRouter()


def get_predictor(request: Request):
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return predictor


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check(request: Request):
    """Check API and model health."""
    predictor = request.app.state.predictor
    is_loaded = predictor is not None and predictor.model is not None
    return HealthResponse(
        status="ok" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_name=predictor.model_name if is_loaded else "none",
    )


@router.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(body: PredictRequest, request: Request):
    """Classify a single message as spam or ham."""
    predictor = get_predictor(request)
    logger.info(f"Predict request | length={len(body.message)}")

    try:
        result = predictor.predict(body.message)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")

    logger.info(f"Result: {result.label} (spam_prob={result.spam_probability:.4f})")
    return PredictResponse(**result.to_dict())


@router.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
def predict_batch(body: BatchPredictRequest, request: Request):
    """Classify a batch of messages (max 100)."""
    predictor = get_predictor(request)
    logger.info(f"Batch predict request | count={len(body.messages)}")

    try:
        results = predictor.predict_batch(body.messages)
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed.")

    spam_count = sum(1 for r in results if r.is_spam)
    return BatchPredictResponse(
        results=[PredictResponse(**r.to_dict()) for r in results],
        total=len(results),
        spam_count=spam_count,
        ham_count=len(results) - spam_count,
    )
