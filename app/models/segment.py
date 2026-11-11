"""
ORM model for an individual transcription segment (timestamp + text).
"""
from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.connection import Base


class TranscriptionSegment(Base):
    __tablename__ = "transcription_segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("transcription_jobs.id", ondelete="CASCADE"), nullable=False
    )
    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    job: Mapped["TranscriptionJob"] = relationship(  # noqa: F821
        "TranscriptionJob", back_populates="segments"
    )

    def __repr__(self) -> str:
        return f"<Segment job={self.job_id} [{self.start_time:.2f}-{self.end_time:.2f}]>"
