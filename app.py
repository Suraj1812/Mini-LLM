import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import WEB_DIR, get_settings
from schemas import (
    ActionResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    StatusResponse,
    TrainRequest,
)
from service import MiniLLMService
from state import AppState


LOGGER = logging.getLogger(__name__)


def create_app():
    settings = get_settings()
    state = AppState(settings=settings)
    state.bootstrap()
    service = MiniLLMService(settings, state)

    app = FastAPI(
        title="Mini LLM",
        version="1.0.0",
        description="Train and run the existing Mini LLM through a simple web UI.",
    )
    app.state.settings = settings
    app.state.app_state = state
    app.state.service = service

    app.add_middleware(GZipMiddleware, minimum_size=512)
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.middleware("http")
    async def guard_requests(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > settings.max_request_bytes:
                    return JSONResponse(
                        {"error": "Request body is too large."},
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    )
            except ValueError:
                return JSONResponse(
                    {"error": "Invalid Content-Length header."},
                    status_code=status.HTTP_400_BAD_REQUEST,
                )

        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        return response

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/health", response_model=HealthResponse)
    async def health():
        snapshot = state.snapshot()
        return {
            "ok": True,
            "status": snapshot["status"],
            "model_ready": snapshot["model_ready"],
        }

    @app.get("/api/status", response_model=StatusResponse)
    async def get_status():
        return state.snapshot()

    @app.post(
        "/api/train",
        response_model=ActionResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def train(payload: TrainRequest):
        text = (payload.text or "").strip()
        if not text and settings.data_path.exists():
            text = settings.data_path.read_text(encoding="utf-8").strip()
        if not text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Training text is required.",
            )
        if len(text) > settings.max_training_chars:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"Training text exceeds the {settings.max_training_chars} character limit."
                ),
            )
        if payload.epochs > settings.max_epochs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Epochs must be between 1 and {settings.max_epochs}.",
            )

        try:
            service.start_training(text, payload.epochs)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc

        return {"message": "Training started.", "status": "running"}

    @app.post("/api/generate", response_model=GenerateResponse)
    async def generate(payload: GenerateRequest):
        if payload.length > settings.max_generation_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Requested generation length exceeds the configured limit "
                    f"of {settings.max_generation_length}."
                ),
            )
        if not state.artifact_paths_ready():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Train the model before generating text.",
            )

        try:
            output = service.generate(payload.prompt, payload.length)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        return {
            "prompt": payload.prompt,
            "output": output,
            "length": payload.length,
        }

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_: Request, exc: Exception):
        LOGGER.exception("Unhandled application error")
        return JSONResponse(
            {"error": "Internal server error."},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    return app


app = create_app()
