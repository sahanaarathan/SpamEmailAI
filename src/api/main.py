import os
import sys
import time
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api.routes.predict import router as predict_router
from src.model.predictor import SpamPredictor
from src.data.preprocessor import TextPreprocessor

# ─── Logging Setup ────────────────────────────────────────────────────────────

def setup_logging(config: dict):
    log_cfg = config.get("logging", {})
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_cfg.get("level", "INFO"),
        format=log_cfg.get(
            "format",
            "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        ),
        colorize=True,
    )
    log_dir = log_cfg.get("dir", "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger.add(
        f"{log_dir}/api.log",
        level=log_cfg.get("level", "INFO"),
        rotation=log_cfg.get("rotation", "10 MB"),
        retention=log_cfg.get("retention", "1 week"),
        format=log_cfg.get("format", "{time} | {level} | {message}"),
    )


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─── App Lifespan ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    setup_logging(config)
    logger.info("Starting Spam Email AI API...")

    preprocessor = TextPreprocessor(
        short_forms=config["preprocessing"].get("short_forms"),
        max_input_length=config["preprocessing"].get("max_input_length", 5000),
    )
    predictor = SpamPredictor(
        model_dir=config["model"]["save_dir"],
        preprocessor=preprocessor,
    )
    try:
        predictor.load()
        app.state.predictor = predictor
        logger.info("Model loaded successfully.")
    except FileNotFoundError as e:
        logger.warning(f"Model not found: {e}. Run 'make train' first.")
        app.state.predictor = None

    yield

    logger.info("Shutting down API.")


# ─── Rate Limiter ─────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)


# ─── App Factory ──────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Spam Email AI",
        description="Production-grade spam detection API powered by ML.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
        return response

    # Global error handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    # Routers
    app.include_router(predict_router, prefix="/api/v1")

    return app


app = create_app()
