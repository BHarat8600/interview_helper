from __future__ import annotations

from pydantic import BaseModel, Field


class ProcessAudioResponse(BaseModel):
    transcription: str = Field(default="")
    answer: str = Field(default="")


class ErrorResponse(BaseModel):
    error: str
    detail: str
    code: str
