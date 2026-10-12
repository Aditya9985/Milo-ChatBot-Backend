# Milo-ChatBot-Backend

FastAPI service for Milo chat and optional text-to-speech. The API does not
store conversation content on the server; the frontend keeps conversation
history on the user's device.

## Deploy on Render

1. Create a **Blueprint** in Render and select this repository. Render reads
   `render.yaml` to create the web service.
2. In the service's **Environment** settings, add `OPENAI_API_KEY` as a secret.
   Never put the key in this repository or in frontend code.
3. Set `CORS_ORIGINS` to the exact deployed web frontend origin(s), separated
   by commas, for example `https://your-app.example.com`. Keep `*` only for
   development or if any browser origin is intentionally allowed. Native
   mobile clients do not use browser CORS.
4. Deploy, then check `https://<your-service>.onrender.com/health`.
5. Set that HTTPS base URL in Milo's frontend **Settings**.

Render injects the listening port through `PORT`; the service binds to
`0.0.0.0` and exposes `/health` as its health check.

The service is stateless and needs no database or persistent disk. This avoids
sharing one user's prompts/history with every other user. Chat requests are not
retained by this backend; the client stores its own local conversation history.

## Run locally

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set OPENAI_API_KEY in .env; do not commit that file.
uvicorn main:app --reload
```

The local API is available at `http://localhost:8000`. Interactive API docs are
at `/docs`.
