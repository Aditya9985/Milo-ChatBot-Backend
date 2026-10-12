import os
from datetime import datetime, timezone
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from openai import APIError, OpenAI
from pydantic import BaseModel, Field, field_validator

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set in the environment.")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
client = OpenAI(api_key=OPENAI_API_KEY)

MILO_SYSTEM_PROMPT = (
    "You are Milo, a friendly and thoughtful AI companion. Be warm, expressive, "
    "and concise, usually answering in one to three natural sentences. Support "
    "Hindi and English, including everyday conversational Hindi and natural "
    "Indian English. Do not impersonate existing fictional characters or claim "
    "to be a person."
)

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

    @field_validator("message")
    @classmethod
    def strip_message(cls, message: str) -> str:
        message = message.strip()
        if not message:
            raise ValueError("Message cannot be empty.")
        return message


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    voice: str = "nova"

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


@app.post("/chat", response_model=MessageResponse)
def chat_endpoint(req: MessageRequest) -> MessageResponse:
    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": MILO_SYSTEM_PROMPT},
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


@app.post("/tts")
def text_to_speech(req: TTSRequest) -> Response:
    try:
        speech = client.audio.speech.create(
            model="tts-1",
            voice=req.voice,
            input=req.text,
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
