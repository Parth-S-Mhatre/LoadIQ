"""SQLite-backed document store for the MVP.

This intentionally uses the standard library only.  It is enough for a single
Render instance; replace it with PostgreSQL + object storage when the feature
moves beyond MVP scale.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from .schemas import DocumentMeta, DocumentStatus, ExtractionResult

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BASE_DIR / "data" / "documents.db"
DEFAULT_STORAGE_DIR = BASE_DIR / "data" / "documents"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


class DocumentStore:
    def __init__(self, db_path: Path | None = None, storage_dir: Path | None = None) -> None:
        self.db_path = Path(os.getenv("DOCUMENTS_DB_PATH", db_path or DEFAULT_DB_PATH))
        self.storage_dir = Path(
            os.getenv("DOCUMENTS_STORAGE_DIR", storage_dir or DEFAULT_STORAGE_DIR)
        )
        self._lock = Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _init_schema(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    status TEXT NOT NULL,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    text_chars INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error TEXT,
                    storage_path TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS extractions (
                    document_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id, created_at)"
            )

    def create_document(
        self,
        *,
        document_id: str,
        user_id: str,
        filename: str,
        content_type: str,
        size_bytes: int,
        sha256: str,
        storage_path: Path,
    ) -> DocumentMeta:
        now = utc_now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    id, user_id, filename, content_type, size_bytes, sha256,
                    status, page_count, text_chars, created_at, updated_at,
                    error, storage_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, NULL, ?)
                """,
                (
                    document_id,
                    user_id,
                    filename,
                    content_type,
                    size_bytes,
                    sha256,
                    "queued",
                    now,
                    now,
                    str(storage_path),
                ),
            )
        return self.get_document(document_id, user_id)  # type: ignore[return-value]

    def get_document(self, document_id: str, user_id: str) -> DocumentMeta | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ? AND user_id = ?",
                (document_id, user_id),
            ).fetchone()
        return self._row_to_meta(row) if row else None

    def get_document_any_user(self, document_id: str) -> DocumentMeta | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return self._row_to_meta(row) if row else None

    def get_storage_path(self, document_id: str) -> Path | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT storage_path FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return Path(row["storage_path"]) if row else None

    def list_documents(self, user_id: str, limit: int = 50) -> list[DocumentMeta]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [self._row_to_meta(row) for row in rows]

    def list_user_extractions(
        self, user_id: str, exclude_document_id: str | None = None, limit: int = 24
    ) -> list[ExtractionResult]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT e.payload
                FROM extractions e
                JOIN documents d ON d.id = e.document_id
                WHERE d.user_id = ? AND d.id != ?
                ORDER BY d.created_at DESC
                LIMIT ?
                """,
                (user_id, exclude_document_id or "", limit),
            ).fetchall()
        results: list[ExtractionResult] = []
        for row in rows:
            try:
                results.append(ExtractionResult.model_validate(json.loads(row["payload"])))
            except Exception:
                continue
        return results

    def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        *,
        page_count: int | None = None,
        text_chars: int | None = None,
        error: str | None = None,
    ) -> None:
        sets = ["status = ?", "updated_at = ?"]
        values: list[Any] = [status, utc_now()]
        if page_count is not None:
            sets.append("page_count = ?")
            values.append(page_count)
        if text_chars is not None:
            sets.append("text_chars = ?")
            values.append(text_chars)
        if error is not None:
            sets.append("error = ?")
            values.append(error)
        values.append(document_id)
        with self._lock, self._connect() as connection:
            connection.execute(
                f"UPDATE documents SET {', '.join(sets)} WHERE id = ?",
                values,
            )

    def save_extraction(self, document_id: str, extraction: ExtractionResult) -> None:
        payload = _json(extraction.model_dump(mode="json"))
        now = utc_now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO extractions (document_id, payload, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (document_id, payload, extraction.created_at, now),
            )

    def get_extraction(self, document_id: str) -> ExtractionResult | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM extractions WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        if not row:
            return None
        try:
            return ExtractionResult.model_validate(json.loads(row["payload"]))
        except Exception:
            return None

    def delete_document(self, document_id: str, user_id: str) -> bool:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT storage_path FROM documents WHERE id = ? AND user_id = ?",
                (document_id, user_id),
            ).fetchone()
            if not row:
                return False
            connection.execute(
                "DELETE FROM documents WHERE id = ? AND user_id = ?",
                (document_id, user_id),
            )
        storage_path = Path(row["storage_path"])
        try:
            storage_path.unlink(missing_ok=True)
        except OSError:
            pass
        return True

    @staticmethod
    def _row_to_meta(row: sqlite3.Row) -> DocumentMeta:
        return DocumentMeta(
            id=row["id"],
            user_id=row["user_id"],
            filename=row["filename"],
            content_type=row["content_type"],
            size_bytes=row["size_bytes"],
            sha256=row["sha256"],
            status=row["status"],
            page_count=row["page_count"],
            text_chars=row["text_chars"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            error=row["error"],
        )


store = DocumentStore()
