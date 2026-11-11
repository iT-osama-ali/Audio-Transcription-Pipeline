"""
Celery worker: picks up transcription jobs from Redis and processes them.

Retry policy:
- Up to MAX_RETRY_ATTEMPTS retries.
- Exponential back-off: delay = RETRY_BACKOFF_BASE_SECONDS * 2^attempt.
- Permanent errors (AudioValidationError) are NOT retried.
"""
import logging
from datetime import datetime, timezone

from celery import Celery
from celery.exceptions import MaxRetriesExceededError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.connection import SessionLocal
from app.models.job import TranscriptionJob
from app.models.segment import TranscriptionSegment
from app.services.audio_processor import AudioValidationError
from app.services.transcriber import transcribe_file

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Celery application
# ---------------------------------------------------------------------------

celery_app = Celery(
    "transcription_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_concurrency=settings.WORKER_CONCURRENCY,
    task_acks_late=True,          # acknowledge only after completion
    task_reject_on_worker_lost=True,
)


# ---------------------------------------------------------------------------
# Helper: DB operations
# ---------------------------------------------------------------------------

def _get_job(db: Session, job_id: str) -> TranscriptionJob | None:
    return db.get(TranscriptionJob, job_id)


def _set_status(db: Session, job: TranscriptionJob, status: str) -> None:
    job.status = status
    job.updated_at = datetime.now(timezone.utc)
    db.commit()


def _save_result(
    db: Session,
    job: TranscriptionJob,
    language: str,
    full_text: str,
    segments: list,
) -> None:
    job.detected_language = language
    job.transcript = full_text
    job.status = "Completed"
    job.updated_at = datetime.now(timezone.utc)

    # Remove old segments in case of a retry
    db.query(TranscriptionSegment).filter(
        TranscriptionSegment.job_id == job.id
    ).delete()

    for seg in segments:
        db.add(
            TranscriptionSegment(
                job_id=job.id,
                start_time=seg.start,
                end_time=seg.end,
                text=seg.text,
            )
        )
    db.commit()


def _record_error(
    db: Session,
    job: TranscriptionJob,
    error: str,
    final: bool,
) -> None:
    job.error_message = error
    job.status = "Failed" if final else "Pending"
    job.updated_at = datetime.now(timezone.utc)
    db.commit()


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="transcription_worker.transcribe_job",
    max_retries=settings.MAX_RETRY_ATTEMPTS,
)
def transcribe_job(self, job_id: str) -> dict:
    """
    Main Celery task: transcribe the audio file for *job_id*.

    Returns a summary dict on success; raises on permanent failure.
    """
    db: Session = SessionLocal()
    try:
        job = _get_job(db, job_id)
        if job is None:
            logger.error("Job %s not found in database.", job_id)
            return {"error": "job_not_found"}

        # Idempotency guard: skip if already completed
        if job.status == "Completed":
            logger.info("Job %s already completed, skipping.", job_id)
            return {"status": "already_completed"}

        logger.info("Processing job %s (attempt %d)…", job_id, self.request.retries + 1)
        _set_status(db, job, "Processing")

        # --- Transcription ---
        result = transcribe_file(job.audio_path)

        _save_result(
            db,
            job,
            language=result.language,
            full_text=result.full_text,
            segments=result.segments,
        )
        logger.info("Job %s completed successfully.", job_id)
        return {"status": "completed", "language": result.language}

    except AudioValidationError as exc:
        # Permanent failure – do not retry
        logger.error("Job %s: permanent audio error: %s", job_id, exc)
        job = _get_job(db, job_id)
        if job:
            job.retry_count = self.request.retries
            _record_error(db, job, str(exc), final=True)
        raise  # Let Celery mark the task as FAILURE without retrying

    except Exception as exc:
        logger.warning("Job %s: transient error: %s", job_id, exc)
        job = _get_job(db, job_id)

        backoff_delay = settings.RETRY_BACKOFF_BASE_SECONDS * (
            2 ** self.request.retries
        )

        if job:
            job.retry_count = self.request.retries + 1
            is_final = job.retry_count >= settings.MAX_RETRY_ATTEMPTS
            _record_error(db, job, str(exc), final=is_final)

        try:
            raise self.retry(exc=exc, countdown=backoff_delay)
        except MaxRetriesExceededError:
            logger.error("Job %s: max retries exceeded.", job_id)
            raise

    finally:
        db.close()
