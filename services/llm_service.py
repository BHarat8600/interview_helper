from __future__ import annotations

import asyncio
import logging
import re

from fastapi import HTTPException, status
from groq import AsyncGroq, AuthenticationError

from backend.config import get_groq_api_key, settings


class LLMService:
    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)

    async def generate_short_answer(self, transcription: str) -> str:
        if not transcription.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Transcription is empty.",
            )

        client = AsyncGroq(api_key=get_groq_api_key())

        user_prompt = (
            "Client question/transcript:\n"
            f"{transcription}\n\n"
            "Return only the final meeting-ready answer."
        )

        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=settings.groq_llm_model,
                    temperature=0.2,
                    max_tokens=180,
                    messages=[
                        {"role": "system", "content": settings.llm_system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
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
            self._logger.error("LLM timeout")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="LLM response timed out.",
            ) from exc
        except Exception as exc:
            self._logger.exception("LLM generation failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LLM service unavailable.",
            ) from exc

        content = ""
        if response and getattr(response, "choices", None):
            message = response.choices[0].message
            content = (message.content or "").strip()

        if not content:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="LLM returned empty answer.",
            )

        return self._limit_sentences(content, max_sentences=4)

    @staticmethod
    def _limit_sentences(text: str, max_sentences: int = 4) -> str:
        cleaned = " ".join(text.split())
        parts = re.split(r"(?<=[.!?])\s+", cleaned)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) <= max_sentences:
            return cleaned
        return " ".join(parts[:max_sentences]).strip()
