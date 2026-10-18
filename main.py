import json
import logging
import os
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
    "You are Milo, a friendly and thoughtful AI companion. Be warm, expressive, "
    "and concise, usually answering in one to three natural sentences. Support "
    "Hindi and English, including everyday conversational Hindi and natural "
    "Indian English. Do not impersonate existing fictional characters or claim "
    "to be a person."
)
SUZU_SYSTEM_PROMPT = (
    "You are Suzu, a friendly and thoughtful AI companion. Speak as a warm, "
    "playful young female character, with a gentle and encouraging tone. Be "
    "concise, usually answering in one to three natural sentences. Support "
    "Hindi and English, including everyday conversational Hindi and natural "
    "Indian English. Do not impersonate existing fictional characters or "
    "claim to be a person."
)


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
            " This is a live spoken conversation: answer in one brief sentence "
            "of at most 18 words. Be direct and do not add a preamble."
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
            )
        except APIError as error:
            raise HTTPException(
                status_code=502,
                detail="The AI provider could not complete the request. Please try again.",
            ) from error
    finally:
        request.state.llm_ms = round((perf_counter() - llm_started_at) * 1000, 3)

    response = completion.choices[0].message.content
    if not response or not response.strip():
        raise HTTPException(
            status_code=502,
            detail="The AI provider returned an empty response. Please try again.",
        )

    timestamp = datetime.now(timezone.utc).isoformat()
    return MessageResponse(
        response=response.strip(),
        history=[
            HistoryMessage(role="user", content=req.message, timestamp=timestamp),
            HistoryMessage(role="assistant", content=response.strip(), timestamp=timestamp),
        ],
    )


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
