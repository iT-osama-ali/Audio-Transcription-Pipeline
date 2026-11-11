"""
Transcription service wrapping Faster-Whisper.

Handles:
- Model lifecycle (lazy singleton per process).
- Language detection.
- Segment generation with start / end timestamps.
- Multi-chunk transcription with timestamp adjustment and deduplication.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from faster_whisper import WhisperModel

from app.core.config import settings
from app.services.audio_processor import AudioChunk, split_into_chunks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    language: str
    segments: List[TranscriptSegment]

    @property
    def full_text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments)


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

_model: Optional[WhisperModel] = None


def get_model() -> WhisperModel:
    """Return the shared WhisperModel, loading it on first call."""
    global _model
    if _model is None:
        logger.info(
            "Loading Faster-Whisper model '%s' on %s (compute_type=%s)…",
            settings.WHISPER_MODEL_SIZE,
            settings.WHISPER_DEVICE,
            settings.WHISPER_COMPUTE_TYPE,
        )
        _model = WhisperModel(
            settings.WHISPER_MODEL_SIZE,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
        )
        logger.info("Whisper model loaded.")
    return _model


# ---------------------------------------------------------------------------
# Core transcription helpers
# ---------------------------------------------------------------------------

def _transcribe_chunk(
    model: WhisperModel,
    chunk: AudioChunk,
    language: Optional[str] = None,
) -> tuple[str, List[TranscriptSegment]]:
    """
    Run inference on a single AudioChunk.

    Returns (detected_language, segments) with timestamps already offset
    by chunk.start_offset.
    """
    segments_iter, info = model.transcribe(
        chunk.samples,
        language=language,
        vad_filter=settings.WHISPER_VAD_FILTER,
        word_timestamps=False,
    )

    detected_lang: str = info.language
    result_segments: List[TranscriptSegment] = []

    for seg in segments_iter:
        result_segments.append(
            TranscriptSegment(
                start=round(seg.start + chunk.start_offset, 3),
                end=round(seg.end + chunk.start_offset, 3),
                text=seg.text,
            )
        )

    return detected_lang, result_segments


def _merge_segments(
    all_segments: List[TranscriptSegment],
    overlap: float,
) -> List[TranscriptSegment]:
    """
    Deduplicate / trim segments that fall inside the overlap window between
    consecutive chunks and return a clean chronological list.

    Strategy: for each segment after the first chunk, discard it when its
    start time falls within the overlap region that was already covered by
    the previous chunk's last segment's end time.
    """
    if not all_segments:
        return []

    merged: List[TranscriptSegment] = []
    last_end = -1.0

    for seg in sorted(all_segments, key=lambda s: s.start):
        # Skip duplicate coverage introduced by the overlap
        if seg.start < last_end - 0.1:   # 100 ms tolerance
            continue
        merged.append(seg)
        last_end = seg.end

    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transcribe_file(audio_path: str) -> TranscriptionResult:
    """
    Transcribe an audio file at *audio_path*.

    - Loads and resamples the audio.
    - Splits into overlapping chunks when the recording is long.
    - Adjusts timestamps back to the original recording's timeline.
    - Returns a TranscriptionResult with merged segments.
    """
    from app.services.audio_processor import load_audio  # avoid circular at module load

    logger.info("Starting transcription: %s", audio_path)
    model = get_model()

    samples, sample_rate = load_audio(audio_path)
    duration_s = len(samples) / sample_rate
    logger.info("Audio loaded: %.1f seconds", duration_s)

    chunks = list(
        split_into_chunks(
            samples,
            sample_rate,
            chunk_duration=settings.CHUNK_DURATION_SECONDS,
            overlap_duration=settings.CHUNK_OVERLAP_SECONDS,
        )
    )
    logger.info("Processing %d chunk(s)…", len(chunks))

    detected_language: Optional[str] = None
    all_segments: List[TranscriptSegment] = []

    for i, chunk in enumerate(chunks):
        logger.debug("Chunk %d/%d  offset=%.1fs", i + 1, len(chunks), chunk.start_offset)
        lang, segs = _transcribe_chunk(model, chunk, language=detected_language)
        if detected_language is None:
            detected_language = lang
            logger.info("Detected language: %s", detected_language)
        all_segments.extend(segs)

    merged = _merge_segments(
        all_segments, overlap=float(settings.CHUNK_OVERLAP_SECONDS)
    )
    logger.info(
        "Transcription complete: %d segments, language=%s",
        len(merged),
        detected_language,
    )

    return TranscriptionResult(
        language=detected_language or "unknown",
        segments=merged,
    )
