from __future__ import annotations

import logging
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from auth import create_access_token, hash_password, verify_password
from config import is_groq_key_configured, settings
from deps import get_current_user
from models.schemas import (
    ChatHistoryResponse,
    ChatItem,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    ProcessAudioResponse,
    TokenResponse,
    UserLoginRequest,
    UserResponse,
    UserSignupRequest,
)
from services.llm_service import LLMService
from services.transcription import TranscriptionService
from storage_csv import (
    CsvUser,
    append_chat_message,
    create_user,
    get_chat_history,
    get_user_by_username,
    init_storage,
)


def configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


configure_logging()
logger = logging.getLogger("backend")

if not settings.groq_api_key:
    logger.warning("GROQ_API_KEY is not configured. Requests using LLM/transcription will fail until it is set.")

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


@app.on_event("startup")
async def startup() -> None:
    init_storage()


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


@app.post("/auth/signup", response_model=UserResponse, responses={400: {"model": ErrorResponse}})
async def signup(payload: UserSignupRequest) -> UserResponse:
    username = payload.username.strip()
    password = payload.password.strip()
    if not username:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username is required")
    if len(password) < 6:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password must be at least 6 characters")

    existing = get_user_by_username(username)
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")

    try:
        password_hash = hash_password(password)
    except Exception as exc:
        logger.exception("Password hashing failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid password format")

    user = create_user(username=username, password_hash=password_hash)
    return UserResponse(id=user.id, username=user.username, created_at=user.created_at)


@app.post("/auth/login", response_model=TokenResponse, responses={401: {"model": ErrorResponse}})
async def login(payload: UserLoginRequest) -> TokenResponse:
    username = payload.username.strip()
    password = payload.password.strip()
    if not username or not password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username and password are required")

    user = get_user_by_username(username)
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token, expires_in = create_access_token(subject=user.username)
    return TokenResponse(
        access_token=token,
        expires_in=expires_in,
        user=UserResponse(id=user.id, username=user.username, created_at=user.created_at),
    )


@app.get("/auth/me", response_model=UserResponse)
async def me(current_user: CsvUser = Depends(get_current_user)) -> UserResponse:
    return UserResponse(id=current_user.id, username=current_user.username, created_at=current_user.created_at)


@app.post("/chat/respond", response_model=ChatResponse, responses={401: {"model": ErrorResponse}})
async def chat_respond(
    payload: ChatRequest,
    current_user: CsvUser = Depends(get_current_user),
) -> ChatResponse:
    if not is_groq_key_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GROQ_API_KEY is missing or invalid in backend/.env.",
        )

    answer = await llm_service.generate_short_answer(payload.message)

    append_chat_message(user_id=current_user.id, role="user", content=payload.message)
    append_chat_message(user_id=current_user.id, role="assistant", content=answer)
    return ChatResponse(answer=answer)


@app.get("/chat/history", response_model=ChatHistoryResponse)
async def chat_history(
    limit: int = 50,
    current_user: CsvUser = Depends(get_current_user),
) -> ChatHistoryResponse:
    safe_limit = min(max(limit, 1), 200)
    rows = get_chat_history(user_id=current_user.id, limit=safe_limit)
    return ChatHistoryResponse(
        items=[ChatItem(id=r.id, role=r.role, content=r.content, created_at=r.created_at) for r in rows]
    )


@app.post(
    "/process-audio",
    response_model=ProcessAudioResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def process_audio(
    audio: UploadFile = File(...),
    current_user: CsvUser = Depends(get_current_user),
) -> ProcessAudioResponse:
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

    append_chat_message(user_id=current_user.id, role="user", content=transcription)
    append_chat_message(user_id=current_user.id, role="assistant", content=answer)

    return ProcessAudioResponse(transcription=transcription, answer=answer)
