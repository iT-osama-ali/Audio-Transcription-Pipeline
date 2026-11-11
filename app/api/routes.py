"""
REST API routes for the transcription pipeline.

Endpoints
---------
POST   /api/transcriptions              Upload audio, create job
GET    /api/transcriptions              List jobs
GET    /api/transcriptions/{id}         Job status
GET    /api/transcriptions/{id}/result  Transcription result
DELETE /api/transcriptions/{id}         Delete job + audio file
GET    /health                          Health check
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.connection import get_db
from app.models.job import TranscriptionJob
from app.models.segment import TranscriptionSegment
from app.schemas.transcription import (
    HealthResponse,
    JobCreateResponse,
    JobListResponse,
    JobStatusResponse,
    SegmentSchema,
    TranscriptionResultResponse,
)
from app.services.audio_processor import (
    AudioValidationError,
    probe_audio,
    validate_extension,
    validate_file_size,
)
from app.services.storage_service import StorageService
from app.workers.transcription_worker import transcribe_job

logger = logging.getLogger(__name__)
router = APIRouter()
storage = StorageService()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    return HealthResponse(status="ok", version=settings.APP_VERSION)


# ---------------------------------------------------------------------------
# Upload & create job
# ---------------------------------------------------------------------------

@router.post(
    "/api/transcriptions",
    response_model=JobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Transcriptions"],
    summary="Upload an audio file and create a transcription job",
)
async def create_transcription(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    db: Session = Depends(get_db),
):
    # --- Read file contents ---
    contents = await file.read()

    # --- Validate extension ---
    try:
        extension = validate_extension(file.filename or "unknown")
    except AudioValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # --- Validate file size ---
    try:
        validate_file_size(len(contents))
    except AudioValidationError as exc:
        raise HTTPException(status_code=413, detail=str(exc))

    # --- Validate audio readability ---
    try:
        probe_audio(contents, file.filename or "unknown")
    except AudioValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # --- Persist to disk ---
    audio_path = storage.save(contents, extension)

    # --- Create DB record ---
    job = TranscriptionJob(
        original_filename=file.filename or "unknown",
        audio_path=audio_path,
        audio_format=extension,
        file_size=len(contents),
        status="Pending",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # --- Dispatch Celery task ---
    transcribe_job.apply_async(args=[job.id])
    logger.info("Job %s queued for audio '%s'.", job.id, file.filename)

    return JobCreateResponse(
        job_id=job.id,
        status=job.status,
        message="Transcription job created successfully.",
    )


# ---------------------------------------------------------------------------
# List jobs
# ---------------------------------------------------------------------------

@router.get(
    "/api/transcriptions",
    response_model=JobListResponse,
    tags=["Transcriptions"],
    summary="List all transcription jobs",
)
def list_transcriptions(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status_filter: Optional[str] = Query(None, alias="status"),
    db: Session = Depends(get_db),
):
    query = db.query(TranscriptionJob).order_by(TranscriptionJob.created_at.desc())
    if status_filter:
        query = query.filter(TranscriptionJob.status == status_filter)

    total = query.count()
    jobs = query.offset(offset).limit(limit).all()

    return JobListResponse(
        jobs=[JobStatusResponse.model_validate(j) for j in jobs],
        total=total,
    )


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------

@router.get(
    "/api/transcriptions/{job_id}",
    response_model=JobStatusResponse,
    tags=["Transcriptions"],
    summary="Retrieve the status of a transcription job",
)
def get_transcription_status(job_id: str, db: Session = Depends(get_db)):
    job = _get_job_or_404(db, job_id)
    return JobStatusResponse.model_validate(job)


# ---------------------------------------------------------------------------
# Transcription result
# ---------------------------------------------------------------------------

@router.get(
    "/api/transcriptions/{job_id}/result",
    response_model=TranscriptionResultResponse,
    tags=["Transcriptions"],
    summary="Retrieve the transcription result for a completed job",
)
def get_transcription_result(job_id: str, db: Session = Depends(get_db)):
    job = _get_job_or_404(db, job_id)

    if job.status not in ("Completed", "Failed"):
        raise HTTPException(
            status_code=409,
            detail=f"Job is not yet complete (status: {job.status}).",
        )

    segments = (
        db.query(TranscriptionSegment)
        .filter(TranscriptionSegment.job_id == job_id)
        .order_by(TranscriptionSegment.start_time)
        .all()
    )

    return TranscriptionResultResponse(
        job_id=job.id,
        status=job.status,
        language=job.detected_language,
        transcript=job.transcript,
        segments=[
            SegmentSchema(start=s.start_time, end=s.end_time, text=s.text)
            for s in segments
        ],
    )


# ---------------------------------------------------------------------------
# Delete job
# ---------------------------------------------------------------------------

@router.delete(
    "/api/transcriptions/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Transcriptions"],
    summary="Delete a transcription job and its associated audio file",
)
def delete_transcription(job_id: str, db: Session = Depends(get_db)):
    job = _get_job_or_404(db, job_id)

    storage.delete(job.audio_path)
    db.delete(job)
    db.commit()
    logger.info("Job %s deleted.", job_id)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_job_or_404(db: Session, job_id: str) -> TranscriptionJob:
    job = db.get(TranscriptionJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job
