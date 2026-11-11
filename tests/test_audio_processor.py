"""
Tests for app.services.audio_processor
"""
import io
import struct
import wave

import numpy as np
import pytest

from app.services.audio_processor import (
    TARGET_SAMPLE_RATE,
    AudioChunk,
    AudioValidationError,
    load_audio,
    probe_audio,
    split_into_chunks,
    validate_extension,
    validate_file_size,
)


# ---------------------------------------------------------------------------
# validate_extension
# ---------------------------------------------------------------------------

class TestValidateExtension:
    def test_accepts_wav(self):
        assert validate_extension("audio.wav") == "wav"

    def test_accepts_mp3(self):
        assert validate_extension("recording.mp3") == "mp3"

    def test_accepts_m4a(self):
        assert validate_extension("clip.m4a") == "m4a"

    def test_accepts_flac(self):
        assert validate_extension("track.flac") == "flac"

    def test_accepts_ogg(self):
        assert validate_extension("file.ogg") == "ogg"

    def test_rejects_mp4(self):
        with pytest.raises(AudioValidationError, match="mp4"):
            validate_extension("video.mp4")

    def test_rejects_txt(self):
        with pytest.raises(AudioValidationError):
            validate_extension("notes.txt")

    def test_rejects_no_extension(self):
        with pytest.raises(AudioValidationError):
            validate_extension("audio_file")

    def test_case_insensitive(self):
        assert validate_extension("AUDIO.WAV") == "wav"


# ---------------------------------------------------------------------------
# validate_file_size
# ---------------------------------------------------------------------------

class TestValidateFileSize:
    def test_accepts_small_file(self):
        validate_file_size(1024)  # should not raise

    def test_accepts_exact_limit(self):
        from app.core.config import settings
        limit = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        validate_file_size(limit)  # should not raise

    def test_rejects_oversized_file(self):
        from app.core.config import settings
        over_limit = (settings.MAX_FILE_SIZE_MB + 1) * 1024 * 1024
        with pytest.raises(AudioValidationError, match="limit"):
            validate_file_size(over_limit)


# ---------------------------------------------------------------------------
# probe_audio
# ---------------------------------------------------------------------------

def _make_wav_bytes(duration_s=0.5, sample_rate=16_000):
    n = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))
    return buf.getvalue()


class TestProbeAudio:
    def test_valid_wav(self):
        probe_audio(_make_wav_bytes(), "test.wav")  # should not raise

    def test_corrupted_file_raises(self):
        with pytest.raises(AudioValidationError):
            probe_audio(b"garbage data 1234", "corrupt.wav")

    def test_empty_file_raises(self):
        with pytest.raises(AudioValidationError):
            probe_audio(b"", "empty.wav")


# ---------------------------------------------------------------------------
# split_into_chunks
# ---------------------------------------------------------------------------

class TestSplitIntoChunks:
    def _make_samples(self, duration_s: float) -> np.ndarray:
        n = int(TARGET_SAMPLE_RATE * duration_s)
        return np.zeros(n, dtype=np.float32)

    def test_short_audio_yields_one_chunk(self):
        samples = self._make_samples(60.0)  # 1 minute
        chunks = list(split_into_chunks(samples, TARGET_SAMPLE_RATE, chunk_duration=360, overlap_duration=5))
        assert len(chunks) == 1
        assert chunks[0].start_offset == 0.0

    def test_long_audio_yields_multiple_chunks(self):
        samples = self._make_samples(400.0)  # ~6.7 min
        chunks = list(split_into_chunks(samples, TARGET_SAMPLE_RATE, chunk_duration=360, overlap_duration=5))
        assert len(chunks) > 1

    def test_first_chunk_offset_is_zero(self):
        samples = self._make_samples(400.0)
        chunks = list(split_into_chunks(samples, TARGET_SAMPLE_RATE, chunk_duration=360, overlap_duration=5))
        assert chunks[0].start_offset == 0.0

    def test_subsequent_chunks_have_positive_offset(self):
        samples = self._make_samples(400.0)
        chunks = list(split_into_chunks(samples, TARGET_SAMPLE_RATE, chunk_duration=360, overlap_duration=5))
        for chunk in chunks[1:]:
            assert chunk.start_offset > 0.0

    def test_chunk_offsets_are_monotonically_increasing(self):
        samples = self._make_samples(1200.0)  # 20 minutes
        chunks = list(split_into_chunks(samples, TARGET_SAMPLE_RATE, chunk_duration=360, overlap_duration=5))
        offsets = [c.start_offset for c in chunks]
        assert offsets == sorted(offsets)

    def test_all_chunks_are_audio_chunk_instances(self):
        samples = self._make_samples(400.0)
        for chunk in split_into_chunks(samples, TARGET_SAMPLE_RATE):
            assert isinstance(chunk, AudioChunk)
