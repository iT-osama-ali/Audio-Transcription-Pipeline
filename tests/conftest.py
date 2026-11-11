"""
Pytest fixtures shared across all test modules.
"""
import io
import struct
import wave
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.connection import Base, get_db
from app.main import app


# ---------------------------------------------------------------------------
# In-memory SQLite for tests
# ---------------------------------------------------------------------------

TEST_DATABASE_URL = "sqlite://"  # pure in-memory


@pytest.fixture(scope="session")
def engine():
    _engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=_engine)
    yield _engine
    Base.metadata.drop_all(bind=_engine)


@pytest.fixture()
def db_session(engine):
    connection = engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture()
def client(db_session):
    """TestClient with the in-memory DB injected and Celery tasks mocked."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    # Mock Celery task so no real Redis is needed during API tests
    with patch("app.api.routes.transcribe_job") as mock_task:
        mock_task.apply_async = MagicMock()
        yield TestClient(app)

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _make_wav_bytes(duration_s: float = 1.0, sample_rate: int = 16_000) -> bytes:
    """Generate a minimal valid WAV file in memory."""
    num_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # Silence (all zeros)
        wf.writeframes(struct.pack(f"<{num_samples}h", *([0] * num_samples)))
    return buf.getvalue()


@pytest.fixture()
def valid_wav_bytes():
    return _make_wav_bytes(duration_s=1.0)


@pytest.fixture()
def long_wav_bytes():
    """WAV long enough to trigger chunking (>CHUNK_DURATION_SECONDS)."""
    return _make_wav_bytes(duration_s=400.0)  # ~6.7 min


@pytest.fixture()
def corrupted_bytes():
    return b"this is not audio content at all"
