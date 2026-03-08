from __future__ import annotations

import asyncio
import logging
import re

from fastapi import HTTPException, status
from groq import AsyncGroq, AuthenticationError

from config import get_groq_api_key, settings


class LLMService:
    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)

    async def generate_short_answer(self, transcription: str, profile_context: str | None = None) -> str:
        if not transcription.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Transcription is empty.",
            )

        client = AsyncGroq(api_key=get_groq_api_key())

        # user_prompt = (
        #     "Client question/transcript:\n"
        #     f"{transcription}\n\n"
        #     "Return only the final meeting-ready answer."
        # )

    ###=============update on 02-28-2026 =========================
        context_block = ""
        if profile_context and profile_context.strip():
            context_block = (
                "Candidate background context (use this only to personalize examples and framing):\n"
                f"{profile_context.strip()}\n\n"
            )

        coding_required = self._is_explicit_coding_question(transcription)
        code_instruction = (
            "The interviewer explicitly asked for coding/implementation. "
            "Include code in markdown fenced blocks with language tags "
            "(for example: ```python, ```sql, ```javascript). "
            "Preserve exact indentation in code. "
            "For coding questions, do not inline code in plain text."
            if coding_required
            else "This is not an explicitly coding request. "
            "Do NOT include any code, pseudocode, SQL snippets, DAX snippets, or fenced code blocks. "
            "Return text-only explanation."
        )

        user_prompt = (
            f"{context_block}"
            "Interview question:\n"
            f"{transcription}\n\n"
            "Answer as if you are explaining to an interviewer. "
            "Keep it concise and interview-ready in 3-4 short lines while fully covering the concept. "
            "Use the candidate background context when provided. "
            f"{code_instruction}"
        )
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=settings.groq_llm_model,
                    temperature=0.2,
                    max_tokens=450,
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

        return self._format_response(content, max_sentences=4)

    @staticmethod
    def _is_explicit_coding_question(text: str) -> bool:
        normalized = f" {text.lower()} "
        explicit_markers = (
            " code ",
            " coding ",
            " implement ",
            " implementation ",
            " write a program ",
            " write a function ",
            " write code ",
            " python ",
            " javascript ",
            " js ",
            " java ",
            " c++ ",
            " c# ",
            " sql ",
            " query ",
            " dax ",
            " script ",
            " algorithm ",
            " pseudocode ",
            "leetcode",
        )
        return any(marker in normalized for marker in explicit_markers)

    @staticmethod
    def _limit_sentences(text: str, max_sentences: int = 4) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        parts = re.split(r"(?<=[.!?])\s+", cleaned)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) <= max_sentences:
            return cleaned
        return " ".join(parts[:max_sentences]).strip()

    @staticmethod
    def _normalize_prose(text: str, max_sentences: int = 4) -> str:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return ""
        normalized = [LLMService._limit_sentences(p, max_sentences=max_sentences) for p in paragraphs]
        return "\n\n".join(normalized).strip()

    @staticmethod
    def _format_response(text: str, max_sentences: int = 4) -> str:
        normalized = text.replace("\r\n", "\n").strip()
        if LLMService._looks_like_unfenced_code(normalized):
            return LLMService._ensure_fenced_code(normalized)

        if "```" not in normalized:
            return LLMService._normalize_prose(normalized, max_sentences=max_sentences)

        parts = normalized.split("```")
        rebuilt: list[str] = []
        for idx, part in enumerate(parts):
            if idx % 2 == 0:
                prose = LLMService._normalize_prose(part, max_sentences=max_sentences)
                if prose:
                    rebuilt.append(prose)
            else:
                code_block = part.strip("\n")
                if code_block:
                    rebuilt.append(f"```{code_block}```")
        return "\n\n".join(rebuilt).strip()

    @staticmethod
    def _looks_like_unfenced_code(text: str) -> bool:
        if not text or "```" in text:
            return False

        lowered = text.lower()
        language_markers = ("python", "sql", "javascript", "js", "nodejs", "typescript", "java", "c++", "cpp")
        has_language = any(marker in lowered for marker in language_markers)
        code_tokens = (
            "def ",
            "class ",
            "import ",
            "select ",
            "from ",
            "where ",
            "function ",
            "const ",
            "let ",
            "var ",
            "=>",
            "{",
            "}",
            ";",
        )
        has_code_tokens = sum(token in lowered for token in code_tokens) >= 2
        return has_language and has_code_tokens

    @staticmethod
    def _ensure_fenced_code(text: str) -> str:
        lowered = text.lower().lstrip()
        lang = ""
        for candidate in ("python", "sql", "javascript", "js", "nodejs", "typescript", "java", "cpp", "c++"):
            if lowered.startswith(candidate):
                lang = "javascript" if candidate in {"js", "nodejs"} else ("cpp" if candidate == "c++" else candidate)
                text = text[len(candidate) :].lstrip(" :\n\t")
                break

        if not lang:
            if any(word in lowered for word in ("select ", "insert ", "update ", "delete ", "create table")):
                lang = "sql"
            elif any(word in lowered for word in ("def ", "import ", "print(", "lambda ")):
                lang = "python"
            elif any(word in lowered for word in ("function ", "const ", "let ", "=>", "console.log")):
                lang = "javascript"

        cleaned = text.strip()
        if not cleaned:
            return ""
        return f"```{lang}\n{cleaned}\n```" if lang else f"```\n{cleaned}\n```"
