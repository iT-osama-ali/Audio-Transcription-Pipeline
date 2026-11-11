"""
ORM model for a transcription job.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.connection import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TranscriptionJob(Base):
    __tablename__ = "transcription_jobs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    audio_path: Mapped[str] = mapped_column(String(512), nullable=False)
    audio_format: Mapped[str] = mapped_column(String(16), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)

    # Job lifecycle
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="Pending")
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Transcription result
    detected_language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    # Relationship
    segments: Mapped[list["TranscriptionSegment"]] = relationship(  # noqa: F821
        "TranscriptionSegment",
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="TranscriptionSegment.start_time",
    )

    def __repr__(self) -> str:
        return f"<TranscriptionJob id={self.id} status={self.status}>"
