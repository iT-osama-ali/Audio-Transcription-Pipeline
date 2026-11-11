"""
Pydantic schemas for request validation and API responses.
"""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared / inner schemas
# ---------------------------------------------------------------------------

class SegmentSchema(BaseModel):
    start: float = Field(..., description="Segment start time in seconds")
    end: float = Field(..., description="Segment end time in seconds")
    text: str = Field(..., description="Transcribed text for this segment")

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Job lifecycle responses
# ---------------------------------------------------------------------------

class JobCreateResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    # The ORM column is `id`; we expose it as `job_id` in the API response.
    job_id: str = Field(validation_alias="id")
    status: str
    original_filename: str
    audio_format: str
    file_size: int
    retry_count: int
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True, "populate_by_name": True}


class JobListResponse(BaseModel):
    jobs: List[JobStatusResponse]
    total: int


# ---------------------------------------------------------------------------
# Transcription result response
# ---------------------------------------------------------------------------

class TranscriptionResultResponse(BaseModel):
    job_id: str = Field(validation_alias="id")
    status: str
    language: Optional[str] = None
    transcript: Optional[str] = None
    segments: List[SegmentSchema] = []

    model_config = {"from_attributes": True, "populate_by_name": True}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
