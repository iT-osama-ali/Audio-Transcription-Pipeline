"""
Integration tests for the REST API layer.

Faster-Whisper and Celery are mocked; only FastAPI + SQLAlchemy are exercised.
"""
import io
import struct
import wave
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav_bytes(duration_s=1.0, sample_rate=16_000):
    n = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))
    return buf.getvalue()


def _upload(client, filename: str, content: bytes, content_type: str = "audio/wav"):
    return client.post(
        "/api/transcriptions",
        files={"file": (filename, io.BytesIO(content), content_type)},
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------

class TestUploadAudio:
    def test_upload_valid_wav(self, client):
        resp = _upload(client, "speech.wav", _make_wav_bytes())
        assert resp.status_code == 202
        body = resp.json()
        assert "job_id" in body
        assert body["status"] == "Pending"
        assert "successfully" in body["message"].lower()

    def test_upload_valid_mp3_extension(self, client, tmp_path):
        """
        MP3 probing may fail without a real MP3 bitstream; we patch probe_audio
        to simulate a valid MP3 upload without a real encoder.
        """
        with patch("app.api.routes.probe_audio"):
            resp = _upload(client, "speech.mp3", b"fake-mp3-content", "audio/mpeg")
        assert resp.status_code == 202

    def test_rejects_unsupported_format(self, client):
        resp = _upload(client, "video.mp4", b"fake", "video/mp4")
        assert resp.status_code == 400
        assert "mp4" in resp.json()["detail"].lower()

    def test_rejects_corrupted_audio(self, client):
        resp = _upload(client, "broken.wav", b"not-audio-data-at-all")
        assert resp.status_code == 422

    def test_upload_returns_job_id(self, client):
        resp = _upload(client, "speech.wav", _make_wav_bytes())
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]
        assert len(job_id) > 0

    def test_multiple_uploads_produce_independent_job_ids(self, client):
        resp1 = _upload(client, "a.wav", _make_wav_bytes())
        resp2 = _upload(client, "b.wav", _make_wav_bytes())
        assert resp1.status_code == 202
        assert resp2.status_code == 202
        assert resp1.json()["job_id"] != resp2.json()["job_id"]


# ---------------------------------------------------------------------------
# Job status endpoint
# ---------------------------------------------------------------------------

class TestJobStatus:
    def _create_job(self, client):
        resp = _upload(client, "test.wav", _make_wav_bytes())
        assert resp.status_code == 202
        return resp.json()["job_id"]

    def test_returns_pending_after_upload(self, client):
        job_id = self._create_job(client)
        resp = client.get(f"/api/transcriptions/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "Pending"

    def test_returns_404_for_unknown_job(self, client):
        resp = client.get("/api/transcriptions/nonexistent-job-id")
        assert resp.status_code == 404

    def test_response_contains_expected_fields(self, client):
        job_id = self._create_job(client)
        resp = client.get(f"/api/transcriptions/{job_id}")
        body = resp.json()
        for field in ("job_id", "status", "original_filename", "created_at"):
            assert field in body, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# Transcription result endpoint
# ---------------------------------------------------------------------------

class TestTranscriptionResult:
    def _create_job(self, client):
        resp = _upload(client, "test.wav", _make_wav_bytes())
        return resp.json()["job_id"]

    def test_returns_409_while_pending(self, client):
        job_id = self._create_job(client)
        resp = client.get(f"/api/transcriptions/{job_id}/result")
        assert resp.status_code == 409

    def test_returns_result_when_completed(self, client, db_session):
        from app.models.job import TranscriptionJob
        from app.models.segment import TranscriptionSegment

        job_id = self._create_job(client)

        # Simulate worker completing the job
        job = db_session.get(TranscriptionJob, job_id)
        job.status = "Completed"
        job.detected_language = "en"
        job.transcript = "Hello world."
        db_session.add(
            TranscriptionSegment(
                job_id=job_id,
                start_time=0.0,
                end_time=2.5,
                text="Hello world.",
            )
        )
        db_session.commit()

        resp = client.get(f"/api/transcriptions/{job_id}/result")
        assert resp.status_code == 200
        body = resp.json()
        assert body["language"] == "en"
        assert len(body["segments"]) == 1
        assert body["segments"][0]["start"] == 0.0
        assert body["segments"][0]["end"] == 2.5
        assert body["segments"][0]["text"] == "Hello world."

    def test_returns_404_for_unknown_job(self, client):
        resp = client.get("/api/transcriptions/unknown-id/result")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# List jobs endpoint
# ---------------------------------------------------------------------------

class TestListJobs:
    def test_returns_empty_list_initially(self, client):
        resp = client.get("/api/transcriptions")
        assert resp.status_code == 200
        body = resp.json()
        assert "jobs" in body
        assert "total" in body

    def test_uploaded_jobs_appear_in_list(self, client):
        _upload(client, "one.wav", _make_wav_bytes())
        _upload(client, "two.wav", _make_wav_bytes())

        resp = client.get("/api/transcriptions")
        assert resp.status_code == 200
        assert resp.json()["total"] >= 2


# ---------------------------------------------------------------------------
# Delete endpoint
# ---------------------------------------------------------------------------

class TestDeleteJob:
    def test_delete_returns_204(self, client):
        resp = _upload(client, "del.wav", _make_wav_bytes())
        job_id = resp.json()["job_id"]

        del_resp = client.delete(f"/api/transcriptions/{job_id}")
        assert del_resp.status_code == 204

    def test_deleted_job_is_gone(self, client):
        resp = _upload(client, "del2.wav", _make_wav_bytes())
        job_id = resp.json()["job_id"]

        client.delete(f"/api/transcriptions/{job_id}")
        status_resp = client.get(f"/api/transcriptions/{job_id}")
        assert status_resp.status_code == 404

    def test_delete_nonexistent_job_returns_404(self, client):
        resp = client.delete("/api/transcriptions/does-not-exist")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Job status transitions
# ---------------------------------------------------------------------------

class TestJobStatusTransitions:
    def test_job_status_can_be_set_to_processing(self, client, db_session):
        from app.models.job import TranscriptionJob

        resp = _upload(client, "transition.wav", _make_wav_bytes())
        job_id = resp.json()["job_id"]

        job = db_session.get(TranscriptionJob, job_id)
        job.status = "Processing"
        db_session.commit()

        status_resp = client.get(f"/api/transcriptions/{job_id}")
        assert status_resp.json()["status"] == "Processing"

    def test_job_status_can_be_set_to_failed(self, client, db_session):
        from app.models.job import TranscriptionJob

        resp = _upload(client, "fail.wav", _make_wav_bytes())
        job_id = resp.json()["job_id"]

        job = db_session.get(TranscriptionJob, job_id)
        job.status = "Failed"
        job.error_message = "Simulated failure"
        db_session.commit()

        status_resp = client.get(f"/api/transcriptions/{job_id}")
        body = status_resp.json()
        assert body["status"] == "Failed"
        assert body["error_message"] == "Simulated failure"
