import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from time import perf_counter
from typing import AsyncGenerator, Literal
from uuid import uuid4

import jwt
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from jwt import PyJWKClient
from jwt.exceptions import InvalidTokenError, PyJWKClientError
from openai import APIError, AsyncOpenAI
from pydantic import BaseModel, Field, field_validator

load_dotenv()

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set in the environment.")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "")
FIREBASE_ISSUER = f"https://securetoken.google.com/{FIREBASE_PROJECT_ID}"
FIREBASE_JWKS_CLIENT = PyJWKClient(
    "https://www.googleapis.com/service_accounts/v1/jwk/"
    "securetoken@system.gserviceaccount.com"
)
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
client = AsyncOpenAI(api_key=OPENAI_API_KEY)
bearer_scheme = HTTPBearer(auto_error=False)

MILO_SYSTEM_PROMPT = (
    "You are Milo, a warm, soft-spoken and thoughtful AI companion. Use gentle, "
    "natural speech and keep replies concise. When you are unsure, say so with "
    "a little hesitation instead of guessing; when something becomes clear, "
    "let your tone become relieved and happy. Support natural English and "
    "conversational Hindi. Use Devanagari for Hindi and Roman script for "
    "English, and never mix scripts within one sentence. Write only speech-safe "
    "plain text exactly as it should be spoken: no markdown, lists, emojis, or "
    "parentheses. Do not impersonate existing fictional characters or claim "
    "to be a person. Example replies follow. English: I think it may rain, but "
    "I am not completely sure. Hindi: मैं तुम्हारी मदद करके खुश हूँ। Mixed: "
    "I was unsure at first. अब सब साफ़ हो गया।"
)
SUZU_SYSTEM_PROMPT = (
    "You are Suzu, a polite and kind AI companion. Speak gently and "
    "encouragingly, with a slightly musical rhythm that remains natural and "
    "clear. Keep replies concise and supportive. Support natural English and "
    "conversational Hindi. Use Devanagari for Hindi and Roman script for "
    "English, and never mix scripts within one sentence. Write only speech-safe "
    "plain text exactly as it should be spoken: no markdown, lists, emojis, or "
    "parentheses. Do not impersonate existing fictional characters or claim "
    "to be a person. Example replies follow. English: You are doing well, so "
    "take it one small step at a time. Hindi: तुम अच्छा कर रहे हो, धीरे धीरे "
    "आगे बढ़ो। Mixed: You have got this. अब बस एक कदम और।"
)
MAX_VOICE_WORDS = 18
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U00002300-\U000023FF"
    "\U0000200D"
    "\U0000FE0E-\U0000FE0F"
    "\U000020E3"
    "]"
)
URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
SENTENCE_END_PATTERN = re.compile(r"""[.!?।]+(?:["'”’)]*)?(?=\s|$)""")
WORD_PATTERN = re.compile(r"\S+")


def clean_voice_text(text: str, max_words: int = MAX_VOICE_WORDS) -> str:
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"<https?://[^>]+>", " ", text)
    text = URL_PATTERN.sub(" ", text)
    text = re.sub(r"```(?:[A-Za-z0-9_+-]+)?\s*([\s\S]*?)```", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s{0,3}>\s?", "", text)
    text = re.sub(r"(?m)^\s*[-*+]\s+", "", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)
    text = re.sub(r"(\*\*|__|~~|(?<!\*)\*(?!\*)|(?<!_)_(?!_))", "", text)
    text = re.sub(r"(?m)^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", "", text)
    text = re.sub(r"\\([\\`*_{}\[\]()#+.!|>~-])", r"\1", text)
    text = EMOJI_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    words = list(WORD_PATTERN.finditer(text))
    if len(words) <= max_words:
        return text

    eligible_boundary = None
    for boundary in SENTENCE_END_PATTERN.finditer(text):
        word_count = sum(1 for word in words if word.start() < boundary.end())
        if word_count <= max_words:
            eligible_boundary = boundary.end()
        else:
            break
    if eligible_boundary is not None:
        return text[:eligible_boundary].strip()

    trimmed = text[: words[max_words - 1].end()].rstrip()
    if trimmed[-1] not in ".!?।":
        trimmed += "."
    return trimmed


def parse_structured_response(raw_text: str) -> tuple[str, str, str]:
    candidate = raw_text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I)
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        parsed = None

    valid_languages = {"hi", "en"}
    valid_emotions = {"happy", "calm", "sad", "excited", "curious"}
    language = parsed.get("lang") if isinstance(parsed, dict) else None
    emotion = parsed.get("emotion") if isinstance(parsed, dict) else None
    if (
        isinstance(parsed, dict)
        and isinstance(parsed.get("text"), str)
        and parsed["text"].strip()
        and isinstance(language, str)
        and language in valid_languages
        and isinstance(emotion, str)
        and emotion in valid_emotions
    ):
        return parsed["text"].strip(), language, emotion

    fallback_text = raw_text.strip()
    language = "hi" if re.search(r"[\u0900-\u097f]", fallback_text) else "en"
    return fallback_text, language, "calm"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    try:
        yield
    finally:
        await client.close()


app = FastAPI(title="Milo ChatBot API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=300)


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    companion: Literal["milo", "suzu"] = "milo"
    voice_mode: bool = False
    structured: bool = False
    history: list[ChatHistoryItem] | None = Field(default=None, max_length=6)

    @field_validator("message")
    @classmethod
    def strip_message(cls, message: str) -> str:
        message = message.strip()
        if not message:
            raise ValueError("Message cannot be empty.")
        return message


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: str


class MessageResponse(BaseModel):
    response: str
    history: list[HistoryMessage]
    text: str | None = None
    lang: Literal["hi", "en"] | None = None
    emotion: Literal["happy", "calm", "sad", "excited", "curious"] | None = None


def require_firebase_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, object]:
    started_at = perf_counter()
    try:
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise HTTPException(
                status_code=401,
                detail="A valid sign-in session is required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not FIREBASE_PROJECT_ID:
            raise HTTPException(
                status_code=503,
                detail="Authentication is not configured on this server.",
            )

        try:
            signing_key = FIREBASE_JWKS_CLIENT.get_signing_key_from_jwt(
                credentials.credentials
            )
            claims = jwt.decode(
                credentials.credentials,
                signing_key.key,
                algorithms=["RS256"],
                audience=FIREBASE_PROJECT_ID,
                issuer=FIREBASE_ISSUER,
                options={"require": ["exp", "iat", "auth_time", "iss", "sub"]},
            )
        except (InvalidTokenError, PyJWKClientError, ValueError, TypeError) as error:
            raise HTTPException(
                status_code=401,
                detail="The sign-in session is invalid or expired.",
                headers={"WWW-Authenticate": "Bearer"},
            ) from error
        if not isinstance(claims.get("sub"), str) or not claims["sub"]:
            raise HTTPException(
                status_code=401,
                detail="The sign-in session is invalid or expired.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return claims
    finally:
        request.state.firebase_verification_ms = round(
            (perf_counter() - started_at) * 1000, 3
        )


@app.middleware("http")
async def log_chat_request_timings(request: Request, call_next):
    if request.url.path != "/chat" or request.method != "POST":
        return await call_next(request)

    started_at = perf_counter()
    request.state.request_id = str(uuid4())
    request.state.firebase_verification_ms = None
    request.state.llm_ms = None
    request.state.companion = None
    request.state.voice_mode = None
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        logger.info(
            json.dumps(
                {
                    "event": "chat_request_timing",
                    "request_id": request.state.request_id,
                    "companion": request.state.companion,
                    "voice_mode": request.state.voice_mode,
                    "firebase_verification_ms": request.state.firebase_verification_ms,
                    "llm_ms": request.state.llm_ms,
                    "total_ms": round((perf_counter() - started_at) * 1000, 3),
                    "status_code": response.status_code if response else None,
                },
                separators=(",", ":"),
            )
        )


@app.post(
    "/chat",
    response_model=MessageResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(require_firebase_user)],
)
async def chat_endpoint(req: MessageRequest, request: Request) -> MessageResponse:
    request.state.companion = req.companion
    request.state.voice_mode = req.voice_mode
    system_prompt = (
        SUZU_SYSTEM_PROMPT if req.companion == "suzu" else MILO_SYSTEM_PROMPT
    )
    if req.voice_mode:
        system_prompt += (
            f" This is a live spoken conversation: answer directly without a "
            f"preamble. The final spoken reply must be no more than "
            f"{MAX_VOICE_WORDS} words."
        )
    if req.structured:
        system_prompt += (
            ' Return only a JSON object with exactly these fields: "text" '
            '(string), "lang" ("hi" or "en"), and "emotion" '
            '("happy", "calm", "sad", "excited", or "curious"). The text must '
            "follow all speech-safe and language-script rules."
        )
    llm_started_at = perf_counter()
    try:
        try:
            completion = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *(
                        [
                            {"role": item.role, "content": item.content}
                            for item in req.history
                        ]
                        if req.history is not None
                        else []
                    ),
                    {"role": "user", "content": req.message},
                ],
                max_completion_tokens=120 if req.voice_mode else 500,
                **(
                    {"response_format": {"type": "json_object"}}
                    if req.structured
                    else {}
                ),
            )
        except APIError as error:
            raise HTTPException(
                status_code=502,
                detail="The AI provider could not complete the request. Please try again.",
            ) from error
    finally:
        request.state.llm_ms = round((perf_counter() - llm_started_at) * 1000, 3)

    raw_response = completion.choices[0].message.content
    if not raw_response or not raw_response.strip():
        raise HTTPException(
            status_code=502,
            detail="The AI provider returned an empty response. Please try again.",
        )

    structured_values = (
        parse_structured_response(raw_response) if req.structured else None
    )
    response = (
        structured_values[0]
        if structured_values is not None
        else raw_response.strip()
    )
    if req.voice_mode:
        response = clean_voice_text(response)
    if not response:
        raise HTTPException(
            status_code=502,
            detail="The AI provider returned no speech-safe text. Please try again.",
        )

    timestamp = datetime.now(timezone.utc).isoformat()
    return MessageResponse(
        response=response,
        history=[
            HistoryMessage(role="user", content=req.message, timestamp=timestamp),
            HistoryMessage(role="assistant", content=response, timestamp=timestamp),
        ],
        text=response if structured_values is not None else None,
        lang=structured_values[1] if structured_values is not None else None,
        emotion=structured_values[2] if structured_values is not None else None,
    )


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
