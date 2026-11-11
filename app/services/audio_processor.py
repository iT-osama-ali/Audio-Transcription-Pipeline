"""
Audio processing service.

Responsibilities:
- Validate uploaded file extensions and actual audio readability.
- Read raw PCM audio as a NumPy float32 array suitable for Faster-Whisper.
- Split long recordings into overlapping chunks for efficient processing.
"""
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Tuple

import av
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass
class AudioChunk:
    """A short audio excerpt with its position in the original recording."""

    samples: np.ndarray      # float32 PCM at 16 kHz
    sample_rate: int         # always 16 000
    start_offset: float      # seconds from the start of the full recording


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = set(settings.ALLOWED_EXTENSIONS)


class AudioValidationError(ValueError):
    """Raised for files that are invalid or unsupported."""


def validate_extension(filename: str) -> str:
    """
    Return the lowercased extension (without leading dot) or raise
    AudioValidationError for unsupported types.
    """
    suffix = Path(filename).suffix.lstrip(".").lower()
    if not suffix:
        raise AudioValidationError(f"File '{filename}' has no extension.")
    if suffix not in ALLOWED_EXTENSIONS:
        raise AudioValidationError(
            f"Unsupported format '.{suffix}'. "
            f"Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )
    return suffix


def validate_file_size(size_bytes: int) -> None:
    """Raise AudioValidationError when the file exceeds the configured limit."""
    limit = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if size_bytes > limit:
        raise AudioValidationError(
            f"File size {size_bytes / 1_048_576:.1f} MB exceeds the "
            f"{settings.MAX_FILE_SIZE_MB} MB limit."
        )


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

TARGET_SAMPLE_RATE = 16_000  # Faster-Whisper expects 16 kHz mono float32


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    """
    Decode an audio file at *path* into a mono 16 kHz float32 NumPy array.

    Returns (samples, sample_rate).
    Raises AudioValidationError when the file cannot be decoded.
    """
    try:
        container = av.open(path)
        audio_stream = next(
            (s for s in container.streams if s.type == "audio"), None
        )
        if audio_stream is None:
            raise AudioValidationError(f"No audio stream found in '{path}'.")

        resampler = av.AudioResampler(
            format="fltp",          # planar float32
            layout="mono",
            rate=TARGET_SAMPLE_RATE,
        )

        raw_frames: List[np.ndarray] = []
        for packet in container.demux(audio_stream):
            for frame in packet.decode():
                resampled = resampler.resample(frame)
                for rf in resampled:
                    # Each plane is a 1-D float32 array
                    raw_frames.append(rf.to_ndarray()[0])

        # Flush the resampler
        for rf in resampler.resample(None):
            raw_frames.append(rf.to_ndarray()[0])

        container.close()

        if not raw_frames:
            raise AudioValidationError(f"Could not decode any audio from '{path}'.")

        samples = np.concatenate(raw_frames).astype(np.float32)
        return samples, TARGET_SAMPLE_RATE

    except AudioValidationError:
        raise
    except Exception as exc:
        raise AudioValidationError(
            f"Failed to decode audio file '{path}': {exc}"
        ) from exc


def probe_audio(file_bytes: bytes, filename: str) -> None:
    """
    Attempt to open audio from raw bytes to confirm it is a readable audio
    file. Raises AudioValidationError if the content cannot be decoded.
    """
    try:
        buf = io.BytesIO(file_bytes)
        container = av.open(buf)
        has_audio = any(s.type == "audio" for s in container.streams)
        container.close()
        if not has_audio:
            raise AudioValidationError(
                f"'{filename}' contains no audio stream."
            )
    except AudioValidationError:
        raise
    except Exception as exc:
        raise AudioValidationError(
            f"'{filename}' could not be opened as audio: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Chunk splitting
# ---------------------------------------------------------------------------

def split_into_chunks(
    samples: np.ndarray,
    sample_rate: int,
    chunk_duration: int = settings.CHUNK_DURATION_SECONDS,
    overlap_duration: int = settings.CHUNK_OVERLAP_SECONDS,
) -> Generator[AudioChunk, None, None]:
    """
    Yield overlapping AudioChunk objects from a full audio array.

    When the total duration is shorter than *chunk_duration* the entire
    recording is yielded as one chunk with start_offset=0.
    """
    total_samples = len(samples)
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    step_samples = chunk_samples - overlap_samples

    if total_samples <= chunk_samples:
        yield AudioChunk(
            samples=samples,
            sample_rate=sample_rate,
            start_offset=0.0,
        )
        return

    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        yield AudioChunk(
            samples=samples[start:end],
            sample_rate=sample_rate,
            start_offset=start / sample_rate,
        )
        if end >= total_samples:
            break
        start += step_samples
