from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import is_groq_key_configured, settings
from backend.models.schemas import ErrorResponse, ProcessAudioResponse
from backend.services.llm_service import LLMService
from backend.services.transcription import TranscriptionService


def configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


configure_logging()
logger = logging.getLogger("backend")

if not settings.groq_api_key:
    logger.warning("GROQ_API_KEY is not configured. Requests will fail until it is set.")

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=[m.strip() for m in settings.cors_allow_methods.split(",") if m.strip()],
    allow_headers=[h.strip() for h in settings.cors_allow_headers.split(",") if h.strip()],
)

transcription_service = TranscriptionService()
llm_service = LLMService()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="validation_error",
            detail="Invalid request payload.",
            code="INVALID_REQUEST",
        ).model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="http_error",
            detail=str(exc.detail),
            code=f"HTTP_{exc.status_code}",
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled server error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_server_error",
            detail="Unexpected server error.",
            code="INTERNAL_ERROR",
        ).model_dump(),
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "service": settings.app_name}


@app.post(
    "/process-audio",
    response_model=ProcessAudioResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def process_audio(audio: UploadFile = File(...)) -> ProcessAudioResponse:
    if not is_groq_key_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GROQ_API_KEY is missing or invalid in backend/.env.",
        )

    if not audio.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio filename is missing.",
        )

    if audio.content_type and not audio.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Please upload an audio file.",
        )

    audio_bytes = await audio.read()
    max_bytes = settings.max_audio_size_mb * 1024 * 1024
    if len(audio_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file is too large. Max allowed: {settings.max_audio_size_mb}MB.",
        )
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty.",
        )

    transcription = await transcription_service.transcribe_audio(audio.filename, audio_bytes)
    answer = await llm_service.generate_short_answer(transcription)

    return ProcessAudioResponse(transcription=transcription, answer=answer)
