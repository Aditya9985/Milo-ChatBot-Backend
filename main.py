import os
from datetime import datetime, timezone
from typing import Literal

import jwt
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from jwt import PyJWKClient
from jwt.exceptions import InvalidTokenError, PyJWKClientError
from openai import APIError, OpenAI
from pydantic import BaseModel, Field, field_validator

load_dotenv()

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
client = OpenAI(api_key=OPENAI_API_KEY)
bearer_scheme = HTTPBearer(auto_error=False)

TTS_INSTRUCTIONS = (
    "Speak as an original youthful male animated companion: warm, soft, bright, "
    "slightly nasal, and emotionally present, but never a caricature or an "
    "imitation of any existing character or actor. Use natural conversational "
    "Indian English and everyday Hindi pronunciation. Keep a consistent voice "
    "identity, relaxed pacing, clear articulation, gentle rises and falls, and "
    "short natural pauses. Sound friendly and sincere, with light warmth; avoid "
    "a robotic, overly formal, deep, or exaggerated delivery."
)

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
SUZU_TTS_INSTRUCTIONS = (
    "Speak as an original youthful female animated companion: warm, bright, "
    "gentle, and emotionally present, but never a caricature or an imitation "
    "of any existing character or actor. Use natural conversational Indian "
    "English and everyday Hindi pronunciation. Keep a consistent voice "
    "identity, relaxed pacing, clear articulation, gentle rises and falls, "
    "and short natural pauses. Sound friendly and sincere; avoid robotic, "
    "overly formal, or exaggerated delivery."
)
COMPANION_SPEECH = {
    "milo": {"name": "Milo", "voice": "ash", "instructions": TTS_INSTRUCTIONS},
    "suzu": {
        "name": "Suzu",
        "voice": "nova",
        "instructions": SUZU_TTS_INSTRUCTIONS,
    },
}

app = FastAPI(title="Milo ChatBot API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    companion: Literal["milo", "suzu"] = "milo"

    @field_validator("message")
    @classmethod
    def strip_message(cls, message: str) -> str:
        message = message.strip()
        if not message:
            raise ValueError("Message cannot be empty.")
        return message


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    companion: Literal["milo", "suzu"] = "milo"

    @field_validator("text")
    @classmethod
    def strip_text(cls, text: str) -> str:
        text = text.strip()
        if not text:
            raise ValueError("Text cannot be empty.")
        return text


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: str


class MessageResponse(BaseModel):
    response: str
    history: list[HistoryMessage]


def require_firebase_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, object]:
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


@app.post(
    "/chat",
    response_model=MessageResponse,
    dependencies=[Depends(require_firebase_user)],
)
def chat_endpoint(req: MessageRequest) -> MessageResponse:
    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        SUZU_SYSTEM_PROMPT
                        if req.companion == "suzu"
                        else MILO_SYSTEM_PROMPT
                    ),
                },
                {"role": "user", "content": req.message},
            ],
            max_completion_tokens=500,
        )
    except APIError as error:
        raise HTTPException(
            status_code=502,
            detail="The AI provider could not complete the request. Please try again.",
        ) from error

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


@app.post("/tts", dependencies=[Depends(require_firebase_user)])
def text_to_speech(req: TTSRequest) -> Response:
    companion = COMPANION_SPEECH[req.companion]
    try:
        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=companion["voice"],
            input=req.text,
            instructions=companion["instructions"],
        )
    except APIError as error:
        raise HTTPException(
            status_code=502,
            detail="The AI provider could not generate speech. Please try again.",
        ) from error
    return Response(content=speech.content, media_type="audio/mpeg")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "provider": "openai", "model": OPENAI_MODEL}
