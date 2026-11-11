"""
Storage service: saves uploaded audio files to the local filesystem with
server-controlled filenames to prevent path traversal.
"""
import logging
import os
import uuid
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)


class StorageService:
    def __init__(self) -> None:
        self._base: Path = settings.upload_path

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def save(self, file_data: bytes, extension: str) -> str:
        """
        Persist *file_data* under a UUID-based filename.

        Returns the absolute path string of the saved file.
        """
        filename = f"{uuid.uuid4()}.{extension.lower().lstrip('.')}"
        dest = self._base / filename

        dest.write_bytes(file_data)
        logger.info("Saved audio file: %s (%d bytes)", dest, len(file_data))
        return str(dest.resolve())

    def delete(self, path: str) -> bool:
        """
        Remove a previously stored file.  Returns True on success.
        """
        target = Path(path)

        # Safety: ensure the target is inside the upload directory
        try:
            target.resolve().relative_to(self._base.resolve())
        except ValueError:
            logger.warning("Refused to delete file outside upload dir: %s", path)
            return False

        if target.exists():
            target.unlink()
            logger.info("Deleted audio file: %s", path)
            return True

        logger.warning("File not found for deletion: %s", path)
        return False

    def exists(self, path: str) -> bool:
        return Path(path).exists()
