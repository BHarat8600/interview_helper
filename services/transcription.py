from __future__ import annotations

import asyncio
import io
import logging

from fastapi import HTTPException, status
from groq import AsyncGroq, AuthenticationError

from config import get_groq_api_key, settings


class TranscriptionService:
    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)

    async def transcribe_audio(self, filename: str, audio_bytes: bytes) -> str:
        if not audio_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty audio payload.",
            )

        client = AsyncGroq(api_key=get_groq_api_key())

        file_obj = io.BytesIO(audio_bytes)
        file_obj.name = filename or "audio.wav"

        try:
            response = await asyncio.wait_for(
                client.audio.transcriptions.create(
                    file=file_obj,
                    model=settings.groq_whisper_model,
                    response_format="verbose_json",
                    language="en",
                    temperature=0,
                ),
                timeout=settings.groq_timeout_seconds,
            )
        except AuthenticationError as exc:
            self._logger.error("Groq authentication failed: invalid API key")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid GROQ_API_KEY. Update backend/.env with a valid key.",
            ) from exc
        except asyncio.TimeoutError as exc:
            self._logger.error("Transcription timeout")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Transcription timed out.",
            ) from exc
        except Exception as exc:
            self._logger.exception("Transcription failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Transcription service unavailable.",
            ) from exc

        text = ""
        if hasattr(response, "text"):
            text = (response.text or "").strip()
        elif isinstance(response, dict):
            text = str(response.get("text", "")).strip()

        if not text:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not transcribe audio.",
            )
        return text
