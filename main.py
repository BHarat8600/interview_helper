from __future__ import annotations

import logging
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from admin_routes import router as admin_router
from auth import create_access_token, decode_token, hash_password, verify_password
from config import BASE_DIR, is_groq_key_configured, settings
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
from storage_admin_csv import (
    append_api_usage,
    append_error_log,
    append_login_log,
    count_user_llm_requests_today,
    ensure_user_control,
    get_user_profile,
    get_user_control,
    init_admin_storage,
    is_api_enabled,
    is_global_llm_enabled,
    is_rag_enabled,
    is_service_active,
    send_admin_alert_email,
)
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

app.include_router(admin_router)


def _to_endpoint_key(path: str, method: str) -> tuple[str, bool]:
    key = f"{method.lower()}:{path}"
    is_llm = False
    if path == "/auth/signup" and method.upper() == "POST":
        key = "auth.signup"
    elif path == "/auth/login" and method.upper() == "POST":
        key = "auth.login"
    elif path == "/auth/me" and method.upper() == "GET":
        key = "auth.me"
    elif path == "/chat/respond" and method.upper() == "POST":
        key = "chat.respond"
        is_llm = True
    elif path == "/chat/history" and method.upper() == "GET":
        key = "chat.history"
    elif path == "/process-audio" and method.upper() == "POST":
        key = "process.audio"
        is_llm = True
    elif path.startswith("/admin/"):
        key = f"admin.{method.lower()}"
    return key, is_llm


def _extract_request_user(request: Request) -> CsvUser | None:
    header = request.headers.get("authorization", "").strip()
    if not header.lower().startswith("bearer "):
        return None
    token = header[7:].strip()
    if not token:
        return None
    try:
        username = decode_token(token)
    except ValueError:
        return None
    return get_user_by_username(username)


def _enforce_user_access(current_user: CsvUser, endpoint_key: str, requires_llm: bool = False) -> None:
    if not is_api_enabled(endpoint_key):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="This API is currently disabled.",
        )

    control = get_user_control(current_user.id)
    if not control.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is disabled.",
        )

    if not requires_llm:
        return

    if not is_service_active():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is currently deactivated.",
        )
    if not is_global_llm_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM is globally disabled.",
        )
    if not control.llm_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="LLM access is disabled for this user.",
        )
    if control.daily_limit > 0 and count_user_llm_requests_today(current_user.id) >= control.daily_limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Daily LLM request limit exceeded.",
        )


def _build_rag_context(user_id: int) -> str | None:
    if not is_rag_enabled():
        return None
    profile = get_user_profile(user_id)
    if profile is None:
        return None

    parts: list[str] = []
    if profile.years_experience:
        parts.append(f"Experience: {profile.years_experience}")
    if profile.skills:
        parts.append(f"Skills: {profile.skills}")
    if profile.technologies:
        parts.append(f"Technologies: {profile.technologies}")
    if profile.projects:
        parts.append(f"Projects: {profile.projects}")
    if not parts:
        return None
    return "\n".join(parts)


@app.on_event("startup")
async def startup() -> None:
    init_storage()
    init_admin_storage()


@app.middleware("http")
async def api_usage_middleware(request: Request, call_next: Any) -> Any:
    endpoint_key, is_llm = _to_endpoint_key(request.url.path, request.method)
    current_user = _extract_request_user(request)
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        try:
            append_api_usage(
                user_id=current_user.id if current_user else None,
                username=current_user.username if current_user else "",
                endpoint_key=endpoint_key,
                path=request.url.path,
                method=request.method,
                status_code=status_code,
                is_llm=is_llm,
            )
        except Exception as exc:
            logger.warning("Failed to append API usage log: %s", exc)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    append_error_log("WARNING", "validation", str(exc))
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
    if exc.status_code >= 500:
        append_error_log("ERROR", "http", str(exc.detail))
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
    append_error_log("ERROR", "unhandled_exception", str(exc))
    send_admin_alert_email(
        subject="Backend critical error",
        body=f"Path: {request.url.path}\nMethod: {request.method}\nError: {str(exc)}",
    )
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


@app.get("/")
async def admin_portal() -> FileResponse:
    return FileResponse(BASE_DIR / "admin_portal.html")


@app.post("/auth/signup", response_model=UserResponse, responses={400: {"model": ErrorResponse}})
async def signup(payload: UserSignupRequest) -> UserResponse:
    if not is_api_enabled("auth.signup"):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="This API is currently disabled.")

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
    ensure_user_control(user.id)
    return UserResponse(id=user.id, username=user.username, created_at=user.created_at)


@app.post("/auth/login", response_model=TokenResponse, responses={401: {"model": ErrorResponse}})
async def login(payload: UserLoginRequest, request: Request) -> TokenResponse:
    if not is_api_enabled("auth.login"):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="This API is currently disabled.")

    username = payload.username.strip()
    password = payload.password.strip()
    if not username or not password:
        append_login_log(user_id=None, username=username, success=False, ip=request.client.host if request.client else "")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username and password are required")

    user = get_user_by_username(username)
    if not user or not verify_password(password, user.password_hash):
        append_login_log(user_id=None, username=username, success=False, ip=request.client.host if request.client else "")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    if not get_user_control(user.id).is_active:
        append_login_log(user_id=user.id, username=username, success=False, ip=request.client.host if request.client else "")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User is disabled.")

    append_login_log(
        user_id=user.id,
        username=username,
        success=True,
        ip=request.client.host if request.client else "",
        user_agent=request.headers.get("user-agent", ""),
    )
    token, expires_in = create_access_token(subject=user.username)
    return TokenResponse(
        access_token=token,
        expires_in=expires_in,
        user=UserResponse(id=user.id, username=user.username, created_at=user.created_at),
    )


@app.get("/auth/me", response_model=UserResponse)
async def me(current_user: CsvUser = Depends(get_current_user)) -> UserResponse:
    _enforce_user_access(current_user=current_user, endpoint_key="auth.me", requires_llm=False)
    return UserResponse(id=current_user.id, username=current_user.username, created_at=current_user.created_at)


@app.post("/chat/respond", response_model=ChatResponse, responses={401: {"model": ErrorResponse}})
async def chat_respond(
    payload: ChatRequest,
    current_user: CsvUser = Depends(get_current_user),
) -> ChatResponse:
    _enforce_user_access(current_user=current_user, endpoint_key="chat.respond", requires_llm=True)

    if not is_groq_key_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GROQ_API_KEY is missing or invalid in backend/.env.",
        )

    answer = await llm_service.generate_short_answer(
        payload.message,
        profile_context=_build_rag_context(current_user.id),
    )

    append_chat_message(user_id=current_user.id, role="user", content=payload.message)
    append_chat_message(user_id=current_user.id, role="assistant", content=answer)
    return ChatResponse(answer=answer)


@app.get("/chat/history", response_model=ChatHistoryResponse)
async def chat_history(
    limit: int = 50,
    current_user: CsvUser = Depends(get_current_user),
) -> ChatHistoryResponse:
    _enforce_user_access(current_user=current_user, endpoint_key="chat.history", requires_llm=False)

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
    _enforce_user_access(current_user=current_user, endpoint_key="process.audio", requires_llm=True)

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
    answer = await llm_service.generate_short_answer(
        transcription,
        profile_context=_build_rag_context(current_user.id),
    )

    append_chat_message(user_id=current_user.id, role="user", content=transcription)
    append_chat_message(user_id=current_user.id, role="assistant", content=answer)

    return ProcessAudioResponse(transcription=transcription, answer=answer)
