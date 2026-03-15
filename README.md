# Mini LLM

Mini LLM is a small end-to-end app for training the current GPT-style toy model on custom text and generating output through a web UI. The backend is exposed as a FastAPI service and ships with local development, Docker, and CI setup.

## What is included

- Web UI for training and inference
- REST API for health, status, training, and generation
- Runtime artifact storage under `artifacts/`
- Environment-based configuration
- Docker support
- API tests for the main backend flows

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python server.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Environment variables

Copy `.env.example` values into your shell or deployment platform as needed.

- `PORT`: Railway-provided port. The app now respects this automatically.
- `MINI_LLM_HOST`: bind address
- `MINI_LLM_PORT`: bind port
- `MINI_LLM_ARTIFACTS_DIR`: where training artifacts are stored
- `MINI_LLM_MAX_EPOCHS`: upper limit for training requests
- `MINI_LLM_MAX_GENERATION_LENGTH`: upper limit for generation requests
- `MINI_LLM_MAX_TRAINING_CHARS`: max training payload size in characters
- `MINI_LLM_MAX_REQUEST_BYTES`: max accepted HTTP request size

## API

- `GET /api/health`
- `GET /api/status`
- `POST /api/train`
- `POST /api/generate`

Example training request:

```json
{
  "text": "Your training corpus goes here",
  "epochs": 3
}
```

Example generation request:

```json
{
  "prompt": "Artificial intelligence",
  "length": 50
}
```

## Docker

```bash
docker build -t mini-llm .
docker run --rm -p 8000:8000 mini-llm
```

## Deploying to Railway

This repo is set up for Railway already:

- `railway.json` defines the start command, restart policy, and `/api/health` healthcheck.
- The server listens on Railway's injected `PORT` automatically.
- If you attach a Railway Volume, the app will automatically use `RAILWAY_VOLUME_MOUNT_PATH` for model artifacts.

Recommended Railway setup:

1. Deploy this repo as a service.
2. Attach a Volume to the service.
3. Use `/app/artifacts` as the mount path, or any absolute path you prefer.
4. Generate a public domain in Railway networking.

Why the volume matters:

- Training outputs like `mini_llm.pt`, `vocab.json`, and `merges.txt` are runtime artifacts.
- Railway service filesystems are ephemeral between deployments, so without a volume your trained model will disappear on redeploy.

After deploy, visit the service domain, train the model once from the UI, and the artifacts will persist on the attached volume.

## Tests

```bash
pytest
```
