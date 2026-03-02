from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from config import settings
from storage_admin_csv import (
    get_last_login_at,
    get_user_control,
    is_api_enabled,
    list_api_controls,
    list_api_usage,
    list_error_logs,
    set_activation_code,
    set_api_enabled,
    set_global_llm_enabled,
    set_service_active,
    set_user_active,
    set_user_daily_limit,
    set_user_llm_enabled,
    verify_activation_code,
)
from storage_csv import list_users


router = APIRouter(prefix="/admin", tags=["admin"])
admin_bearer = HTTPBearer(auto_error=False)


class AdminLoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=128)


class AdminTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserStatusUpdate(BaseModel):
    is_active: bool


class UserLlmUpdate(BaseModel):
    llm_enabled: bool


class UserLimitUpdate(BaseModel):
    daily_limit: int = Field(ge=0, le=100000)


class GlobalLlmUpdate(BaseModel):
    enabled: bool


class ServiceActivationUpdate(BaseModel):
    service_active: bool
    activation_code: str = Field(min_length=4, max_length=64)
    new_activation_code: str | None = Field(default=None, min_length=4, max_length=64)


class ApiControlUpdate(BaseModel):
    is_enabled: bool


def _create_admin_token(subject: str) -> tuple[str, int]:
    expires_delta = timedelta(minutes=settings.admin_jwt_expire_minutes)
    expires_at = datetime.now(UTC) + expires_delta
    payload = {
        "sub": subject,
        "role": "admin",
        "exp": expires_at,
        "iat": datetime.now(UTC),
    }
    token = jwt.encode(payload, settings.admin_jwt_secret_key, algorithm=settings.jwt_algorithm)
    return token, int(expires_delta.total_seconds())


def _decode_admin_token(token: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(token, settings.admin_jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise ValueError("Invalid admin token") from exc
    if payload.get("role") != "admin":
        raise ValueError("Not an admin token")
    return payload


async def require_admin(
    credentials: HTTPAuthorizationCredentials | None = Depends(admin_bearer),
) -> dict[str, Any]:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing admin access token")
    try:
        return _decode_admin_token(credentials.credentials)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired admin token") from exc


@router.post("/login", response_model=AdminTokenResponse)
async def admin_login(payload: AdminLoginRequest) -> AdminTokenResponse:
    username = payload.username.strip()
    password = payload.password.strip()
    if username != settings.admin_username or password != settings.admin_password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin credentials")

    token, expires_in = _create_admin_token(subject=username)
    return AdminTokenResponse(access_token=token, expires_in=expires_in)


@router.get("/users")
async def admin_list_users(_: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
    users = list_users()
    usage_rows = list_api_usage(limit=2000)
    items: list[dict[str, Any]] = []
    for user in users:
        control = get_user_control(user.id)
        user_usage = [row for row in usage_rows if row.get("user_id") == str(user.id)]
        api_usage_count = len(user_usage)
        llm_request_count = sum(1 for row in user_usage if row.get("is_llm", "").lower() == "true")
        items.append(
            {
                "id": user.id,
                "username": user.username,
                "created_at": user.created_at,
                "last_login_at": get_last_login_at(user.id),
                "is_active": control.is_active,
                "llm_enabled": control.llm_enabled,
                "daily_limit": control.daily_limit,
                "api_usage_count": api_usage_count,
                "llm_request_count": llm_request_count,
            }
        )
    return {"items": items}


@router.patch("/users/{user_id}/status")
async def admin_update_user_status(
    user_id: int,
    payload: UserStatusUpdate,
    _: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    control = set_user_active(user_id=user_id, is_active=payload.is_active)
    return {"user_id": control.user_id, "is_active": control.is_active, "updated_at": control.updated_at}


@router.patch("/users/{user_id}/llm")
async def admin_update_user_llm(
    user_id: int,
    payload: UserLlmUpdate,
    _: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    control = set_user_llm_enabled(user_id=user_id, llm_enabled=payload.llm_enabled)
    return {"user_id": control.user_id, "llm_enabled": control.llm_enabled, "updated_at": control.updated_at}


@router.patch("/users/{user_id}/limit")
async def admin_update_user_limit(
    user_id: int,
    payload: UserLimitUpdate,
    _: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    control = set_user_daily_limit(user_id=user_id, daily_limit=payload.daily_limit)
    return {"user_id": control.user_id, "daily_limit": control.daily_limit, "updated_at": control.updated_at}


@router.patch("/settings/llm-global")
async def admin_update_global_llm(
    payload: GlobalLlmUpdate,
    _: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    set_global_llm_enabled(enabled=payload.enabled)
    return {"llm_global_enabled": payload.enabled}


@router.patch("/settings/service-activation")
async def admin_update_service_activation(
    payload: ServiceActivationUpdate,
    _: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    if not verify_activation_code(payload.activation_code):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid activation code")
    set_service_active(active=payload.service_active)
    if payload.new_activation_code:
        set_activation_code(payload.new_activation_code)
    return {"service_active": payload.service_active}


@router.get("/apis")
async def admin_list_api_controls(_: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
    return {"items": list_api_controls()}


@router.patch("/apis/{endpoint_key:path}")
async def admin_update_api_control(
    endpoint_key: str,
    payload: ApiControlUpdate,
    _: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    key = endpoint_key.strip("/")
    set_api_enabled(endpoint_key=key, is_enabled=payload.is_enabled)
    return {"endpoint_key": key, "is_enabled": is_api_enabled(key)}


@router.get("/logs/api")
async def admin_api_logs(
    limit: int = 100,
    _: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    return {"items": list_api_usage(limit=limit)}


@router.get("/logs/errors")
async def admin_error_logs(
    limit: int = 100,
    _: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    return {"items": list_error_logs(limit=limit)}


@router.get("/stats")
async def admin_stats(_: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
    users = list_users()
    controls = {u.id: get_user_control(u.id) for u in users}
    usage = list_api_usage(limit=2000)
    return {
        "total_users": len(users),
        "active_users": sum(1 for u in users if controls[u.id].is_active),
        "llm_enabled_users": sum(1 for u in users if controls[u.id].llm_enabled),
        "total_api_requests": len(usage),
        "total_llm_requests": sum(1 for r in usage if r.get("is_llm", "").lower() == "true"),
    }
