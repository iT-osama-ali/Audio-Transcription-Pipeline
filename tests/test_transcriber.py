"""
Tests for app.services.transcriber

All Faster-Whisper calls are mocked so no model download is required.
"""
import io
import struct
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.transcriber import (
    TranscriptSegment,
    TranscriptionResult,
    _merge_segments,
    transcribe_file,
)


# ---------------------------------------------------------------------------
# _merge_segments
# ---------------------------------------------------------------------------

class TestMergeSegments:
    def _seg(self, start: float, end: float, text: str = "x") -> TranscriptSegment:
        return TranscriptSegment(start=start, end=end, text=text)

    def test_empty_input_returns_empty(self):
        assert _merge_segments([], overlap=5.0) == []

    def test_non_overlapping_segments_preserved(self):
        segs = [self._seg(0, 2), self._seg(2, 4), self._seg(4, 6)]
        result = _merge_segments(segs, overlap=0.0)
        assert len(result) == 3

    def test_duplicate_overlap_removed(self):
        # Simulate overlap: second chunk re-produces segments 8-10 s
        segs = [
            self._seg(0.0, 5.0, "first"),
            self._seg(5.0, 10.0, "second"),
            # These two are within the overlap of the second chunk start
            self._seg(8.0, 10.0, "overlap duplicate"),
            self._seg(10.0, 15.0, "third"),
        ]
        result = _merge_segments(segs, overlap=5.0)
        texts = [s.text for s in result]
        assert "overlap duplicate" not in texts
        assert "third" in texts

    def test_output_is_chronological(self):
        segs = [self._seg(10, 12), self._seg(0, 2), self._seg(5, 7)]
        result = _merge_segments(segs, overlap=0.0)
        starts = [s.start for s in result]
        assert starts == sorted(starts)

    def test_single_segment_returned_as_is(self):
        segs = [self._seg(0, 3, "hello")]
        result = _merge_segments(segs, overlap=0.0)
        assert len(result) == 1
        assert result[0].text == "hello"


# ---------------------------------------------------------------------------
# transcribe_file (mocked Whisper)
# ---------------------------------------------------------------------------

def _make_wav_file(tmp_path, duration_s=1.0):
    """Write a minimal WAV to a temp file and return its path."""
    n = int(16_000 * duration_s)
    path = tmp_path / "test.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16_000)
        wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))
    return str(path)


def _mock_segment(start, end, text):
    seg = MagicMock()
    seg.start = start
    seg.end = end
    seg.text = text
    return seg


def _make_mock_model(segments, language="en"):
    info = MagicMock()
    info.language = language

    model = MagicMock()
    model.transcribe.return_value = (iter(segments), info)
    return model


class TestTranscribeFile:
    def test_returns_transcription_result(self, tmp_path):
        wav = _make_wav_file(tmp_path)
        mock_segs = [_mock_segment(0.0, 2.0, "Hello world.")]
        model = _make_mock_model(mock_segs)

        with patch("app.services.transcriber.get_model", return_value=model):
            result = transcribe_file(wav)

        assert isinstance(result, TranscriptionResult)
        assert result.language == "en"
        assert len(result.segments) == 1
        assert result.segments[0].text == "Hello world."

    def test_segments_have_correct_timestamps(self, tmp_path):
        wav = _make_wav_file(tmp_path)
        mock_segs = [
            _mock_segment(0.0, 3.25, "Hello, welcome to the meeting."),
            _mock_segment(3.25, 6.80, "Today we will discuss the project."),
        ]
        model = _make_mock_model(mock_segs)

        with patch("app.services.transcriber.get_model", return_value=model):
            result = transcribe_file(wav)

        assert result.segments[0].start == 0.0
        assert result.segments[0].end == 3.25
        assert result.segments[1].start == 3.25
        assert result.segments[1].end == 6.80

    def test_full_text_joins_segments(self, tmp_path):
        wav = _make_wav_file(tmp_path)
        mock_segs = [
            _mock_segment(0.0, 2.0, "Hello."),
            _mock_segment(2.0, 4.0, "World."),
        ]
        model = _make_mock_model(mock_segs)

        with patch("app.services.transcriber.get_model", return_value=model):
            result = transcribe_file(wav)

        assert "Hello." in result.full_text
        assert "World." in result.full_text

    def test_language_detection(self, tmp_path):
        wav = _make_wav_file(tmp_path)
        model = _make_mock_model([], language="fr")

        with patch("app.services.transcriber.get_model", return_value=model):
            result = transcribe_file(wav)

        assert result.language == "fr"

    def test_long_audio_timestamps_offset_correctly(self, tmp_path):
        """
        For a file longer than CHUNK_DURATION_SECONDS the second chunk's
        timestamps must be offset so they represent positions in the full
        recording, not just within the chunk.
        """
        from app.core.config import settings

        # Build a WAV longer than one chunk
        duration = settings.CHUNK_DURATION_SECONDS + 60
        wav = _make_wav_file(tmp_path, duration_s=duration)

        # First chunk: segment at 0-5 s
        # Second chunk: Whisper would report 0-5 s within the chunk
        # → after offset they should appear at ~CHUNK_DURATION_SECONDS seconds
        first_segs = [_mock_segment(0.0, 5.0, "First chunk text.")]
        second_segs = [_mock_segment(0.0, 5.0, "Second chunk text.")]

        call_count = 0

        def side_effect(audio, **kwargs):
            nonlocal call_count
            info = MagicMock()
            info.language = "en"
            if call_count == 0:
                call_count += 1
                return iter(first_segs), info
            else:
                call_count += 1
                return iter(second_segs), info

        model = MagicMock()
        model.transcribe.side_effect = side_effect

        with patch("app.services.transcriber.get_model", return_value=model):
            result = transcribe_file(wav)

        # The second segment's start should be offset by at least
        # CHUNK_DURATION_SECONDS - CHUNK_OVERLAP_SECONDS
        min_offset = settings.CHUNK_DURATION_SECONDS - settings.CHUNK_OVERLAP_SECONDS
        second_chunk_segs = [s for s in result.segments if s.start >= min_offset]
        assert len(second_chunk_segs) > 0, (
            "Expected at least one segment from the second chunk with an offset timestamp"
        )
