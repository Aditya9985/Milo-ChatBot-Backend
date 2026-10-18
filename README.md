# Milo-ChatBot-Backend

FastAPI service for authenticated Milo chat and natural speech generation.
Firebase Authentication manages sign-in and sessions; the API validates
Firebase ID tokens.
Conversation content is not stored by the backend.

## Deploy on Render

1. Create a **Blueprint** in Render and select this repository. Render reads
   `render.yaml` to create the web service.
2. The `.python-version` file pins the service to Python 3.12.10, which is
   compatible with the pinned Pydantic dependencies.
3. Add `OPENAI_API_KEY` as a Render secret. Never put it in this repository or
   in frontend code.
4. Create a Firebase project and enable **Authentication > Google**, then
   copy its **Project ID** from Project settings. Set `FIREBASE_PROJECT_ID` in
   Render. The backend uses Google's public Firebase signing keys to verify
   ID tokens; no service-account key or auth database is required.
5. Set `CORS_ORIGINS` to the exact deployed web frontend origin(s), separated
   by commas, for example `https://your-app.example.com`. Keep `*` only for
   development or if any browser origin is intentionally allowed. Native
   mobile clients do not use browser CORS.
6. Deploy, then check `https://<your-service>.onrender.com/health`.
7. The Flutter app uses the generated `frontend/lib/firebase_options.dart`.
   Re-run `flutterfire configure --project=milo-2107d --platforms=android`
   from the frontend directory after changing Firebase apps or adding
   platforms.

8. Set the backend HTTPS base URL in Milo's frontend **Settings** if it is not
   already set.

Render injects the listening port through `PORT`; the service binds to
`0.0.0.0` and exposes `/health` as its health check.

The service is stateless and needs no database or persistent disk. Chat
requests are not retained by this backend; the client stores its own local
conversation history. `/chat` requires a valid Firebase ID token in the
`Authorization: Bearer` header. `/health` remains public.

Chat uses the asynchronous OpenAI client so provider requests do not block
FastAPI's event loop. The Flutter app speaks responses with the device's
native text-to-speech engine; no backend speech-generation request is made.
Each `/chat` request emits one structured timing log with a request ID,
companion, voice-mode flag, Firebase verification time, LLM time, and total
request time. User message text is not included in these logs. `/health`
returns `{"status":"ok"}` without authentication or an AI provider call.
Chat clients may optionally send up to six prior `user`/`assistant` messages
in `history`, with each content value limited to 300 characters. System
messages are not accepted from clients.

## Run locally

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set OPENAI_API_KEY and FIREBASE_PROJECT_ID in .env.
# Do not commit .env.
uvicorn main:app --reload
```

The local API is available at `http://localhost:8000`. Interactive API docs are
at `/docs`.
