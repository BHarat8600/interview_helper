from __future__ import annotations

import csv
import hmac
import smtplib
from dataclasses import dataclass
from datetime import UTC, datetime
from email.message import EmailMessage
from threading import Lock

from config import BASE_DIR, settings
from storage_csv import list_users


DATA_DIR = BASE_DIR / "data"
USER_CONTROLS_CSV = DATA_DIR / "user_controls.csv"
API_USAGE_CSV = DATA_DIR / "api_usage.csv"
ERROR_LOGS_CSV = DATA_DIR / "error_logs.csv"
LOGIN_LOGS_CSV = DATA_DIR / "login_logs.csv"
GLOBAL_SETTINGS_CSV = DATA_DIR / "global_settings.csv"
API_CONTROLS_CSV = DATA_DIR / "api_controls.csv"
USER_PROFILES_CSV = DATA_DIR / "user_profiles.csv"

_lock = Lock()

DEFAULT_SETTINGS = {
    "llm_global_enabled": "true",
    "rag_enabled": "false",
    "service_active": "true",
    "service_activation_code": "8668317759",
    "alert_email": "stalinade05@gmail.com",
}

DEFAULT_API_CONTROLS = [
    "auth.signup",
    "auth.login",
    "auth.me",
    "chat.respond",
    "chat.history",
    "process.audio",
]


@dataclass
class UserControl:
    user_id: int
    is_active: bool
    llm_enabled: bool
    daily_limit: int
    updated_at: datetime


@dataclass
class UserProfile:
    user_id: int
    years_experience: str
    skills: str
    technologies: str
    projects: str
    updated_at: datetime


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _as_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_datetime_safe(value: str) -> datetime:
    raw = (value or "").strip()
    if not raw:
        return datetime.now(UTC)
    try:
        return _parse_datetime(raw)
    except Exception:
        return datetime.now(UTC)


def init_admin_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not USER_CONTROLS_CSV.exists():
        with USER_CONTROLS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["user_id", "is_active", "llm_enabled", "daily_limit", "updated_at"])
            writer.writeheader()
    if not API_USAGE_CSV.exists():
        with API_USAGE_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "id",
                    "user_id",
                    "username",
                    "endpoint_key",
                    "path",
                    "method",
                    "status_code",
                    "is_llm",
                    "created_at",
                ],
            )
            writer.writeheader()
    if not ERROR_LOGS_CSV.exists():
        with ERROR_LOGS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "level", "source", "message", "created_at"])
            writer.writeheader()
    if not LOGIN_LOGS_CSV.exists():
        with LOGIN_LOGS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "user_id", "username", "success", "ip", "user_agent", "created_at"],
            )
            writer.writeheader()
    if not GLOBAL_SETTINGS_CSV.exists():
        with GLOBAL_SETTINGS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["key", "value", "updated_at"])
            writer.writeheader()
            now = _now_iso()
            for key, value in DEFAULT_SETTINGS.items():
                writer.writerow({"key": key, "value": value, "updated_at": now})
    if not API_CONTROLS_CSV.exists():
        with API_CONTROLS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["endpoint_key", "is_enabled", "updated_at"])
            writer.writeheader()
            now = _now_iso()
            for endpoint_key in DEFAULT_API_CONTROLS:
                writer.writerow({"endpoint_key": endpoint_key, "is_enabled": "true", "updated_at": now})
    if not USER_PROFILES_CSV.exists():
        with USER_PROFILES_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["user_id", "years_experience", "skills", "technologies", "projects", "updated_at"],
            )
            writer.writeheader()


def _read_user_controls() -> list[UserControl]:
    rows: list[UserControl] = []
    if not USER_CONTROLS_CSV.exists():
        return rows
    with USER_CONTROLS_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                UserControl(
                    user_id=int(r["user_id"]),
                    is_active=_as_bool(r["is_active"], default=True),
                    llm_enabled=_as_bool(r["llm_enabled"], default=True),
                    daily_limit=max(int(r.get("daily_limit", "100")), 0),
                    updated_at=_parse_datetime(r["updated_at"]),
                )
            )
    return rows


def _write_user_controls(rows: list[UserControl]) -> None:
    with USER_CONTROLS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["user_id", "is_active", "llm_enabled", "daily_limit", "updated_at"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "user_id": str(row.user_id),
                    "is_active": "true" if row.is_active else "false",
                    "llm_enabled": "true" if row.llm_enabled else "false",
                    "daily_limit": str(row.daily_limit),
                    "updated_at": row.updated_at.isoformat(),
                }
            )


def ensure_user_control(user_id: int) -> UserControl:
    with _lock:
        rows = _read_user_controls()
        for row in rows:
            if row.user_id == user_id:
                return row
        new_row = UserControl(
            user_id=user_id,
            is_active=True,
            llm_enabled=True,
            daily_limit=100,
            updated_at=datetime.now(UTC),
        )
        rows.append(new_row)
        _write_user_controls(rows)
        return new_row


def get_user_control(user_id: int) -> UserControl:
    return ensure_user_control(user_id)


def set_user_active(user_id: int, is_active: bool) -> UserControl:
    with _lock:
        rows = _read_user_controls()
        now = datetime.now(UTC)
        for idx, row in enumerate(rows):
            if row.user_id == user_id:
                rows[idx] = UserControl(
                    user_id=row.user_id,
                    is_active=is_active,
                    llm_enabled=row.llm_enabled,
                    daily_limit=row.daily_limit,
                    updated_at=now,
                )
                _write_user_controls(rows)
                return rows[idx]
        created = UserControl(user_id=user_id, is_active=is_active, llm_enabled=True, daily_limit=100, updated_at=now)
        rows.append(created)
        _write_user_controls(rows)
        return created


def set_user_llm_enabled(user_id: int, llm_enabled: bool) -> UserControl:
    with _lock:
        rows = _read_user_controls()
        now = datetime.now(UTC)
        for idx, row in enumerate(rows):
            if row.user_id == user_id:
                rows[idx] = UserControl(
                    user_id=row.user_id,
                    is_active=row.is_active,
                    llm_enabled=llm_enabled,
                    daily_limit=row.daily_limit,
                    updated_at=now,
                )
                _write_user_controls(rows)
                return rows[idx]
        created = UserControl(user_id=user_id, is_active=True, llm_enabled=llm_enabled, daily_limit=100, updated_at=now)
        rows.append(created)
        _write_user_controls(rows)
        return created


def set_user_daily_limit(user_id: int, daily_limit: int) -> UserControl:
    with _lock:
        rows = _read_user_controls()
        now = datetime.now(UTC)
        safe_limit = max(daily_limit, 0)
        for idx, row in enumerate(rows):
            if row.user_id == user_id:
                rows[idx] = UserControl(
                    user_id=row.user_id,
                    is_active=row.is_active,
                    llm_enabled=row.llm_enabled,
                    daily_limit=safe_limit,
                    updated_at=now,
                )
                _write_user_controls(rows)
                return rows[idx]
        created = UserControl(user_id=user_id, is_active=True, llm_enabled=True, daily_limit=safe_limit, updated_at=now)
        rows.append(created)
        _write_user_controls(rows)
        return created


def list_user_controls() -> list[UserControl]:
    with _lock:
        rows = _read_user_controls()
        by_user_id = {r.user_id: r for r in rows}
        users = list_users()
        for user in users:
            if user.id not in by_user_id:
                by_user_id[user.id] = UserControl(
                    user_id=user.id,
                    is_active=True,
                    llm_enabled=True,
                    daily_limit=100,
                    updated_at=datetime.now(UTC),
                )
        full = list(by_user_id.values())
        full.sort(key=lambda x: x.user_id)
        return full


def append_api_usage(
    user_id: int | None,
    username: str | None,
    endpoint_key: str,
    path: str,
    method: str,
    status_code: int,
    is_llm: bool,
) -> None:
    with _lock:
        next_id = 1
        if API_USAGE_CSV.exists():
            with API_USAGE_CSV.open("r", newline="", encoding="utf-8") as rf:
                reader = csv.DictReader(rf)
                for row in reader:
                    next_id = max(next_id, int(row["id"]) + 1)
        with API_USAGE_CSV.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "id",
                    "user_id",
                    "username",
                    "endpoint_key",
                    "path",
                    "method",
                    "status_code",
                    "is_llm",
                    "created_at",
                ],
            )
            writer.writerow(
                {
                    "id": str(next_id),
                    "user_id": "" if user_id is None else str(user_id),
                    "username": username or "",
                    "endpoint_key": endpoint_key,
                    "path": path,
                    "method": method.upper(),
                    "status_code": str(status_code),
                    "is_llm": "true" if is_llm else "false",
                    "created_at": _now_iso(),
                }
            )


def list_api_usage(limit: int = 200) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not API_USAGE_CSV.exists():
        return rows
    with _lock:
        with API_USAGE_CSV.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [dict(r) for r in reader]
    rows.sort(key=lambda r: (r.get("created_at", ""), int(r.get("id", "0"))), reverse=True)
    return rows[: max(1, min(limit, 2000))]


def count_user_llm_requests_today(user_id: int) -> int:
    if not API_USAGE_CSV.exists():
        return 0
    today = datetime.now(UTC).date()
    total = 0
    with _lock:
        with API_USAGE_CSV.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("user_id") != str(user_id):
                    continue
                if not _as_bool(row.get("is_llm", ""), default=False):
                    continue
                created_at = row.get("created_at", "")
                if not created_at:
                    continue
                if _parse_datetime(created_at).date() == today:
                    total += 1
    return total


def append_error_log(level: str, source: str, message: str) -> None:
    with _lock:
        next_id = 1
        if ERROR_LOGS_CSV.exists():
            with ERROR_LOGS_CSV.open("r", newline="", encoding="utf-8") as rf:
                reader = csv.DictReader(rf)
                for row in reader:
                    next_id = max(next_id, int(row["id"]) + 1)
        with ERROR_LOGS_CSV.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "level", "source", "message", "created_at"])
            writer.writerow(
                {
                    "id": str(next_id),
                    "level": level.upper(),
                    "source": source,
                    "message": message,
                    "created_at": _now_iso(),
                }
            )


def list_error_logs(limit: int = 200) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not ERROR_LOGS_CSV.exists():
        return rows
    with _lock:
        with ERROR_LOGS_CSV.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [dict(r) for r in reader]
    rows.sort(key=lambda r: (r.get("created_at", ""), int(r.get("id", "0"))), reverse=True)
    return rows[: max(1, min(limit, 2000))]


def append_login_log(
    user_id: int | None,
    username: str,
    success: bool,
    ip: str = "",
    user_agent: str = "",
) -> None:
    with _lock:
        next_id = 1
        if LOGIN_LOGS_CSV.exists():
            with LOGIN_LOGS_CSV.open("r", newline="", encoding="utf-8") as rf:
                reader = csv.DictReader(rf)
                for row in reader:
                    next_id = max(next_id, int(row["id"]) + 1)
        with LOGIN_LOGS_CSV.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "user_id", "username", "success", "ip", "user_agent", "created_at"],
            )
            writer.writerow(
                {
                    "id": str(next_id),
                    "user_id": "" if user_id is None else str(user_id),
                    "username": username,
                    "success": "true" if success else "false",
                    "ip": ip,
                    "user_agent": user_agent,
                    "created_at": _now_iso(),
                }
            )


def get_last_login_at(user_id: int) -> str | None:
    if not LOGIN_LOGS_CSV.exists():
        return None
    latest: str | None = None
    with _lock:
        with LOGIN_LOGS_CSV.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("user_id") != str(user_id):
                    continue
                if not _as_bool(row.get("success", ""), default=False):
                    continue
                created_at = row.get("created_at", "")
                if not created_at:
                    continue
                if latest is None or created_at > latest:
                    latest = created_at
    return latest


def _read_settings() -> dict[str, str]:
    values = dict(DEFAULT_SETTINGS)
    if not GLOBAL_SETTINGS_CSV.exists():
        return values
    with GLOBAL_SETTINGS_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get("key", "").strip()
            if not key:
                continue
            values[key] = row.get("value", "").strip()
    return values


def get_setting(key: str, default: str = "") -> str:
    with _lock:
        return _read_settings().get(key, default)


def set_setting(key: str, value: str) -> None:
    with _lock:
        settings_map = _read_settings()
        settings_map[key] = value
        now = _now_iso()
        with GLOBAL_SETTINGS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["key", "value", "updated_at"])
            writer.writeheader()
            for each_key in sorted(settings_map.keys()):
                writer.writerow({"key": each_key, "value": settings_map[each_key], "updated_at": now})


def is_global_llm_enabled() -> bool:
    return _as_bool(get_setting("llm_global_enabled", "true"), default=True)


def set_global_llm_enabled(enabled: bool) -> None:
    set_setting("llm_global_enabled", "true" if enabled else "false")


def is_service_active() -> bool:
    return _as_bool(get_setting("service_active", "true"), default=True)


def set_service_active(active: bool) -> None:
    set_setting("service_active", "true" if active else "false")


def verify_activation_code(activation_code: str) -> bool:
    configured = get_setting("service_activation_code", DEFAULT_SETTINGS["service_activation_code"])
    return hmac.compare_digest(configured.strip(), activation_code.strip())


def set_activation_code(activation_code: str) -> None:
    set_setting("service_activation_code", activation_code.strip())


def _read_api_controls() -> dict[str, bool]:
    values = {key: True for key in DEFAULT_API_CONTROLS}
    if not API_CONTROLS_CSV.exists():
        return values
    with API_CONTROLS_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            endpoint_key = row.get("endpoint_key", "").strip()
            if not endpoint_key:
                continue
            values[endpoint_key] = _as_bool(row.get("is_enabled", "true"), default=True)
    return values


def list_api_controls() -> list[dict[str, str]]:
    with _lock:
        rows = _read_api_controls()
    return [{"endpoint_key": k, "is_enabled": v} for k, v in sorted(rows.items())]


def is_api_enabled(endpoint_key: str) -> bool:
    with _lock:
        rows = _read_api_controls()
    return rows.get(endpoint_key, True)


def set_api_enabled(endpoint_key: str, is_enabled: bool) -> None:
    with _lock:
        values = _read_api_controls()
        values[endpoint_key] = is_enabled
        now = _now_iso()
        with API_CONTROLS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["endpoint_key", "is_enabled", "updated_at"])
            writer.writeheader()
            for key in sorted(values.keys()):
                writer.writerow(
                    {"endpoint_key": key, "is_enabled": "true" if values[key] else "false", "updated_at": now}
                )


def send_admin_alert_email(subject: str, body: str) -> bool:
    recipient = (settings.admin_alert_email or get_setting("alert_email", DEFAULT_SETTINGS["alert_email"])).strip()
    if not recipient:
        return False
    if not settings.smtp_host or not settings.smtp_port:
        return False

    sender = (settings.smtp_from_email or settings.smtp_username or "").strip()
    if not sender:
        return False

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient
    message.set_content(body)

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as smtp:
            if settings.smtp_use_tls:
                smtp.starttls()
            if settings.smtp_username and settings.smtp_password:
                smtp.login(settings.smtp_username, settings.smtp_password)
            smtp.send_message(message)
    except Exception:
        return False
    return True


def _read_user_profiles() -> list[UserProfile]:
    rows: list[UserProfile] = []
    if not USER_PROFILES_CSV.exists():
        return rows
    with USER_PROFILES_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r.get("user_id", "").strip():
                continue
            rows.append(
                UserProfile(
                    user_id=int(r["user_id"]),
                    years_experience=r.get("years_experience", "").strip(),
                    skills=r.get("skills", "").strip(),
                    technologies=r.get("technologies", "").strip(),
                    projects=r.get("projects", "").strip(),
                    updated_at=_parse_datetime_safe(r.get("updated_at", "")),
                )
            )
    return rows


def _write_user_profiles(rows: list[UserProfile]) -> None:
    with USER_PROFILES_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["user_id", "years_experience", "skills", "technologies", "projects", "updated_at"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "user_id": str(row.user_id),
                    "years_experience": row.years_experience,
                    "skills": row.skills,
                    "technologies": row.technologies,
                    "projects": row.projects,
                    "updated_at": row.updated_at.isoformat(),
                }
            )


def get_user_profile(user_id: int) -> UserProfile | None:
    with _lock:
        rows = _read_user_profiles()
    for row in rows:
        if row.user_id == user_id:
            return row
    return None


def list_user_profiles() -> list[UserProfile]:
    with _lock:
        rows = _read_user_profiles()
    rows.sort(key=lambda x: x.user_id)
    return rows


def upsert_user_profile(
    user_id: int,
    years_experience: str,
    skills: str,
    technologies: str,
    projects: str,
) -> UserProfile:
    with _lock:
        rows = _read_user_profiles()
        now = datetime.now(UTC)
        for idx, row in enumerate(rows):
            if row.user_id == user_id:
                updated = UserProfile(
                    user_id=user_id,
                    years_experience=years_experience.strip(),
                    skills=skills.strip(),
                    technologies=technologies.strip(),
                    projects=projects.strip(),
                    updated_at=now,
                )
                rows[idx] = updated
                _write_user_profiles(rows)
                return updated
        created = UserProfile(
            user_id=user_id,
            years_experience=years_experience.strip(),
            skills=skills.strip(),
            technologies=technologies.strip(),
            projects=projects.strip(),
            updated_at=now,
        )
        rows.append(created)
        _write_user_profiles(rows)
        return created


def delete_user_profile(user_id: int) -> bool:
    with _lock:
        rows = _read_user_profiles()
        filtered = [row for row in rows if row.user_id != user_id]
        deleted = len(filtered) != len(rows)
        if deleted:
            _write_user_profiles(filtered)
    return deleted


def is_rag_enabled() -> bool:
    return _as_bool(get_setting("rag_enabled", "false"), default=False)


def set_rag_enabled(enabled: bool) -> None:
    set_setting("rag_enabled", "true" if enabled else "false")
