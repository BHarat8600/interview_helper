from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from config import BASE_DIR


DATA_DIR = BASE_DIR / "data"
USERS_CSV = DATA_DIR / "users.csv"
CHATS_CSV = DATA_DIR / "chat_history.csv"

_lock = Lock()


@dataclass
class CsvUser:
    id: int
    username: str
    password_hash: str
    created_at: datetime


@dataclass
class CsvChatItem:
    id: int
    user_id: int
    role: str
    content: str
    created_at: datetime


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def init_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not USERS_CSV.exists():
        with USERS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "username", "password_hash", "created_at"])
            writer.writeheader()
    if not CHATS_CSV.exists():
        with CHATS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "user_id", "role", "content", "created_at"])
            writer.writeheader()


def _read_users() -> list[CsvUser]:
    if not USERS_CSV.exists():
        return []
    rows: list[CsvUser] = []
    with USERS_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                CsvUser(
                    id=int(r["id"]),
                    username=r["username"],
                    password_hash=r["password_hash"],
                    created_at=_parse_datetime(r["created_at"]),
                )
            )
    return rows


def get_user_by_username(username: str) -> CsvUser | None:
    with _lock:
        for user in _read_users():
            if user.username == username:
                return user
    return None


def create_user(username: str, password_hash: str) -> CsvUser:
    with _lock:
        users = _read_users()
        next_id = max((u.id for u in users), default=0) + 1
        created_at = _now_iso()
        with USERS_CSV.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "username", "password_hash", "created_at"])
            writer.writerow(
                {
                    "id": str(next_id),
                    "username": username,
                    "password_hash": password_hash,
                    "created_at": created_at,
                }
            )
        return CsvUser(
            id=next_id,
            username=username,
            password_hash=password_hash,
            created_at=_parse_datetime(created_at),
        )


def append_chat_message(user_id: int, role: str, content: str) -> None:
    with _lock:
        rows = list_chat_messages_all()
        next_id = max((r.id for r in rows), default=0) + 1
        with CHATS_CSV.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "user_id", "role", "content", "created_at"])
            writer.writerow(
                {
                    "id": str(next_id),
                    "user_id": str(user_id),
                    "role": role,
                    "content": content,
                    "created_at": _now_iso(),
                }
            )


def list_chat_messages_all() -> list[CsvChatItem]:
    if not CHATS_CSV.exists():
        return []
    rows: list[CsvChatItem] = []
    with CHATS_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                CsvChatItem(
                    id=int(r["id"]),
                    user_id=int(r["user_id"]),
                    role=r["role"],
                    content=r["content"],
                    created_at=_parse_datetime(r["created_at"]),
                )
            )
    return rows


def get_chat_history(user_id: int, limit: int) -> list[CsvChatItem]:
    with _lock:
        rows = [r for r in list_chat_messages_all() if r.user_id == user_id]
        rows.sort(key=lambda x: (x.created_at, x.id))
        return rows[:limit]
