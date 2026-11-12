# Audio Transcription Pipeline

A backend service that accepts audio files, transcribes speech to text using **Faster-Whisper**, and returns timestamped segments via a REST API.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Key Design Decisions](#key-design-decisions)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Running the Application](#running-the-application)
- [Running Tests](#running-tests)
- [API Reference](#api-reference)
- [Configuration Reference](#configuration-reference)
- [Limitations and Next Steps](#limitations-and-next-steps)

---

## Architecture Overview

```
Client
  │
  ▼
FastAPI (HTTP layer)
  │  validates extension, file size, audio readability
  │  persists job record (SQLite via SQLAlchemy)
  │  saves audio file to local uploads/ directory
  │
  ▼
Redis (message broker)
  │
  ▼
Celery Worker (background processing)
  │  loads audio → resamples to 16 kHz mono float32 (PyAV/FFmpeg)
  │  splits long files into overlapping chunks
  │  calls Faster-Whisper per chunk
  │  adjusts timestamps by chunk offset
  │  merges and deduplicates segments
  │  persists results to SQLite
  │
  ▼
SQLite (job metadata + transcript segments)
```

**Processing flow:**

1. `POST /api/transcriptions` — file uploaded, validated, stored; job created as `Pending`; Celery task queued; response returned immediately.
2. Celery worker picks up the task, sets status to `Processing`, runs Faster-Whisper.
3. On success: status → `Completed`, segments written to DB.
4. On failure: retried up to 3 times with exponential back-off; permanent audio errors skip retries and go straight to `Failed`.
5. Client polls `GET /api/transcriptions/{id}` until `Completed`, then fetches `GET /api/transcriptions/{id}/result`.

**Concurrent uploads:**

Every uploaded file is saved under a server-generated UUID filename, so there are no collisions between simultaneous uploads. Each upload creates one independent job row in the database, meaning uploads never block each other at the API layer. The number of jobs processed in parallel is controlled by `WORKER_CONCURRENCY`. The main concurrency bottleneck in this MVP is SQLite, which serializes writes — switching `DATABASE_URL` to PostgreSQL removes that constraint for high-throughput deployments.

---

## Key Design Decisions

### Why Faster-Whisper?
OpenAI Whisper is accurate but slow on CPU. Faster-Whisper is a CTranslate2-based reimplementation that is 2–4× faster at the same accuracy, works without a GPU, and does not require a paid API.

### Why chunk-based processing?
Loading multi-hour recordings fully into memory is impractical. The file is decoded once, then yielded in overlapping windows (default 6 min, 5 s overlap). The overlap prevents missed words at boundaries; duplicated segments are discarded during merging.

### Why Celery + Redis?
HTTP requests must return immediately regardless of audio length. Celery decouples transcription from the API, allows horizontal worker scaling, and provides built-in retry primitives with exponential back-off.

### Why SQLite for the MVP?
SQLite requires zero infrastructure, makes the project easy to run locally, and is sufficient for the assessment workload. The SQLAlchemy ORM layer means switching to PostgreSQL requires only a one-line `DATABASE_URL` change.

### Why local file storage?
Same rationale as SQLite — zero external dependencies for the MVP. The `StorageService` class is the only place that touches the filesystem, so swapping it for an S3/Azure Blob implementation is straightforward.

### Security considerations
- File extensions are validated against an allowlist.
- Audio readability is verified with PyAV before the file is stored.
- Uploaded files are stored under server-generated UUID filenames (no user-controlled paths).
- Path traversal is blocked in `StorageService.delete()` by resolving relative to the upload root.
- Internal storage paths are never exposed in API responses.

---

## Technology Stack

| Component           | Technology         |
|---------------------|--------------------|
| Language            | Python 3.11+       |
| API framework       | FastAPI            |
| Speech recognition  | Faster-Whisper     |
| Whisper model       | `small` (default)  |
| Audio decoding      | PyAV / FFmpeg      |
| Database            | SQLite             |
| ORM                 | SQLAlchemy 2.x     |
| Background tasks    | Celery             |
| Message broker      | Redis              |
| Validation          | Pydantic v2        |
| Testing             | Pytest             |

---

## Project Structure

```
audio-transcription-pipeline/
│
├── app/
│   ├── main.py                      # FastAPI app factory + startup hook
│   ├── api/
│   │   └── routes.py                # All REST endpoints
│   ├── core/
│   │   └── config.py                # Pydantic Settings (reads .env)
│   ├── services/
│   │   ├── audio_processor.py       # Validation, decoding, chunking
│   │   ├── transcriber.py           # Faster-Whisper wrapper
│   │   └── storage_service.py       # File persistence
│   ├── workers/
│   │   └── transcription_worker.py  # Celery app + task
│   ├── models/
│   │   ├── job.py                   # TranscriptionJob ORM model
│   │   └── segment.py               # TranscriptionSegment ORM model
│   ├── schemas/
│   │   └── transcription.py         # Pydantic request/response schemas
│   └── database/
│       └── connection.py            # Engine, session factory, Base
│
├── uploads/                         # Audio file storage (gitignored)
├── tests/
│   ├── conftest.py                  # Shared fixtures, in-memory DB
│   ├── test_audio_processor.py      # Unit tests for audio layer
│   ├── test_transcriber.py          # Unit tests for transcription layer
│   └── test_api.py                  # Integration tests for REST API
│
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Setup Instructions

### Prerequisites

- Python 3.11 or higher
- [FFmpeg](https://ffmpeg.org/download.html) installed and on `PATH` (required by PyAV)
- Redis server (local install or Docker)

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/iT-osama-ali/Audio-Transcription-Pipeline.git
cd Audio-Transcription-Pipeline
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux
```

Edit `.env` as needed. Defaults work for local development.

### 4. Start Redis

```bash
# Using Docker (simplest)
docker run -d -p 6379:6379 redis:7.4-alpine

# Or install Redis locally: https://redis.io/docs/getting-started/
```

---

## Running the Application

You need **two** processes running simultaneously.

### API server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API is now available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

### Celery worker

In a second terminal (with the same virtual environment activated):

```bash
celery -A app.workers.transcription_worker.celery_app worker --loglevel=info
```

### Docker Compose (all-in-one)

```bash
docker-compose up --build
```

This starts Redis, the API server, and a Celery worker together.

---

## Running Tests

Tests use an in-memory SQLite database and mock all external services (Faster-Whisper, Celery/Redis). No model download or Redis instance is required.

```bash
pytest tests/ -v
```

To run a specific test file:

```bash
pytest tests/test_api.py -v
pytest tests/test_audio_processor.py -v
pytest tests/test_transcriber.py -v
```

---

## API Reference

### Health check

```
GET /health
```

**Response:**
```json
{ "status": "ok", "version": "1.0.0" }
```

---

### Upload audio and create a job

```
POST /api/transcriptions
Content-Type: multipart/form-data
Body: file=<audio file>
```

**Supported formats:** `wav`, `mp3`, `m4a`, `flac`, `ogg`

**Response `202 Accepted`:**
```json
{
  "job_id": "a12b34cd-5678-90ef-abcd-1234567890ab",
  "status": "Pending",
  "message": "Transcription job created successfully."
}
```

**Error responses:**
- `400` — unsupported file extension
- `413` — file exceeds size limit
- `422` — file cannot be decoded as audio

---

### Get job status

```
GET /api/transcriptions/{job_id}
```

**Response `200`:**
```json
{
  "job_id": "a12b34cd-5678-90ef-abcd-1234567890ab",
  "status": "Processing",
  "original_filename": "meeting.wav",
  "audio_format": "wav",
  "file_size": 2048000,
  "retry_count": 0,
  "error_message": null,
  "created_at": "2026-11-12T10:00:00Z",
  "updated_at": "2026-11-12T10:00:05Z"
}
```

**Job status values:** `Pending` → `Processing` → `Completed` / `Failed`

---

### Get transcription result

```
GET /api/transcriptions/{job_id}/result
```

Only available once status is `Completed` or `Failed`.

**Response `200`:**
```json
{
  "job_id": "a12b34cd-5678-90ef-abcd-1234567890ab",
  "status": "Completed",
  "language": "en",
  "transcript": "Hello, welcome to the meeting. Today we will discuss the project.",
  "segments": [
    { "start": 0.0,  "end": 3.25, "text": "Hello, welcome to the meeting." },
    { "start": 3.25, "end": 6.80, "text": "Today we will discuss the project." }
  ]
}
```

---

### List transcription jobs

```
GET /api/transcriptions?limit=20&offset=0&status=Completed
```

**Response `200`:**
```json
{
  "jobs": [ ... ],
  "total": 42
}
```

---

### Delete a job

```
DELETE /api/transcriptions/{job_id}
```

Deletes the job record, all segments, and the uploaded audio file.

**Response:** `204 No Content`

---

## Configuration Reference

All settings are read from environment variables or a `.env` file.

| Variable                   | Default                      | Description                              |
|----------------------------|------------------------------|------------------------------------------|
| `WHISPER_MODEL_SIZE`       | `small`                      | `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3` |
| `WHISPER_DEVICE`           | `cpu`                        | `cpu` or `cuda`                          |
| `WHISPER_COMPUTE_TYPE`     | `int8`                       | `int8`, `float16`, `float32`             |
| `WHISPER_VAD_FILTER`       | `true`                       | Enable Voice Activity Detection          |
| `MAX_FILE_SIZE_MB`         | `500`                        | Maximum upload size in megabytes         |
| `CHUNK_DURATION_SECONDS`   | `360`                        | Chunk length for long audio (6 min)      |
| `CHUNK_OVERLAP_SECONDS`    | `5`                          | Overlap between chunks                   |
| `DATABASE_URL`             | `sqlite:///./transcription.db` | SQLAlchemy database URL                |
| `CELERY_BROKER_URL`        | `redis://localhost:6379/0`   | Celery broker                            |
| `CELERY_RESULT_BACKEND`    | `redis://localhost:6379/1`   | Celery result backend                    |
| `WORKER_CONCURRENCY`       | `2`                          | Celery worker processes                  |
| `MAX_RETRY_ATTEMPTS`       | `3`                          | Max retries for transient failures       |
| `RETRY_BACKOFF_BASE_SECONDS` | `60`                       | Base delay for exponential back-off      |

---

## Limitations and Next Steps

**Single-machine storage**
SQLite and the local `uploads/` directory only work on one machine. A multi-instance deployment needs PostgreSQL (one `DATABASE_URL` change via SQLAlchemy) and an object store like S3 or Azure Blob Storage (one `StorageService` swap).

**Jobs stuck in Processing after a worker crash**
The worker sets `task_acks_late = True`, so a task is re-queued if the worker dies mid-job. However, if the worker crashes after acknowledging but before updating the DB, the job stays in `Processing` indefinitely. A periodic "reaper" task that resets stale `Processing` jobs back to `Pending` would close this gap.

**No manual re-queue endpoint**
Once a job reaches `Failed` there is no API endpoint to retry it. Currently the only options are submitting a new upload or updating the DB directly. A `POST /api/transcriptions/{id}/retry` endpoint would be a clean addition.

**No authentication**
All endpoints are open. This is intentional for the MVP, but any public deployment needs at minimum an API key header or an OAuth 2.0 layer in front of the routes.
