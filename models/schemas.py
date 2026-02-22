from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    error: str
    detail: str
    code: str


class UserSignupRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=128)


class UserLoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=128)


class UserResponse(BaseModel):
    id: int
    username: str
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ChatItem(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime


class ChatResponse(BaseModel):
    answer: str


class ChatHistoryResponse(BaseModel):
    items: list[ChatItem]


class ProcessAudioResponse(BaseModel):
    transcription: str = Field(default="")
    answer: str = Field(default="")
