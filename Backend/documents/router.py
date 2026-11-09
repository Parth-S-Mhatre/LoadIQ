"""FastAPI routes for the document-intelligence MVP."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from .analytics import build_analytics
from .extractor import extract_pdf
from .ocr import OCRConfig, field_recovery_enabled, min_text_chars, ocr_enabled, ocr_mode
from .report import render_report
from .schemas import (
    AnalyticsResult,
    ConfirmRequest,
    DocumentMeta,
    ExtractionResult,
)
from .store import store, utc_now

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_UPLOAD_BYTES = int(os.getenv("DOCUMENTS_MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))


@router.get("/ocr/health")
def ocr_health() -> dict[str, Any]:
    """Report OCR configuration without exposing the API key."""
    config = OCRConfig.from_env()
    enabled = ocr_enabled()
    return {
        "enabled": enabled,
        "configured": config.configured,
        "provider": "nemotron-ocr-v2" if enabled else "none",
        "mode": ocr_mode(),
        "min_text_chars": min_text_chars(),
        "field_recovery": field_recovery_enabled(),
        "endpoint": config.url,
        "api_key_configured": bool(config.api_key),
        "dpi": config.dpi,
        "max_pages": config.max_pages,
        "max_base64_chars": config.max_base64_chars,
    }


def _firebase_user(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token.")
    token = authorization.split(" ", 1)[1].strip()

    try:
        import firebase_admin
        from firebase_admin import auth, credentials
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="Firebase auth mode is enabled but firebase-admin is not installed.",
        ) from exc

    if not firebase_admin._apps:
        service_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        if service_json:
            credential = credentials.Certificate(json.loads(service_json))
        else:
            credential = credentials.ApplicationDefault()
        firebase_admin.initialize_app(credential)

    try:
        decoded = auth.verify_id_token(token)
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired token.") from exc

    return str(decoded["uid"])


def get_user_id(
    x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    """Resolve a tenant for document access.

    DOCUMENTS_AUTH_MODE:
      - none    → X-User-ID or "anonymous" (local development only)
      - header  → X-User-ID is required
      - firebase→ Authorization: Bearer <Firebase ID token>
    """
    mode = os.getenv("DOCUMENTS_AUTH_MODE", "none").strip().lower()

    if mode == "firebase":
        return _firebase_user(authorization)
    if mode == "header":
        if not x_user_id:
            raise HTTPException(status_code=401, detail="X-User-ID header is required.")
        return x_user_id
    return x_user_id or "anonymous"


def _require_document(document_id: str, user_id: str) -> DocumentMeta:
    document = store.get_document(document_id, user_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")
    return document


def _require_extraction(document_id: str, user_id: str) -> ExtractionResult:
    _require_document(document_id, user_id)
    extraction = store.get_extraction(document_id)
    if not extraction:
        raise HTTPException(
            status_code=409,
            detail="Extraction is not ready yet. Poll the document status endpoint.",
        )
    return extraction


def _process_document(document_id: str) -> None:
    document = store.get_document_any_user(document_id)
    if not document:
        return

    storage_path = store.get_storage_path(document_id)
    if not storage_path:
        store.update_status(document_id, "failed", error="Uploaded file record disappeared.")
        return

    store.update_status(document_id, "processing")
    try:
        extraction = extract_pdf(document_id, storage_path)
        store.save_extraction(document_id, extraction)
        store.update_status(
            document_id,
            extraction.status,
            page_count=extraction.page_count,
            text_chars=extraction.text_chars,
        )
    except Exception as exc:  # noqa: BLE001 - surface a useful error to the client
        store.update_status(document_id, "failed", error=str(exc)[:800])


@router.post("", response_model=DocumentMeta, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="Electricity bill PDF")],
    user_id: Annotated[str, Depends(get_user_id)],
) -> DocumentMeta:
    content_type = (file.content_type or "").lower()
    if content_type not in {"application/pdf", "application/x-pdf", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="Only PDF uploads are supported in the MVP.")

    document_id = uuid.uuid4().hex
    destination = store.storage_dir / f"{document_id}.pdf"
    digest = hashlib.sha256()
    size = 0

    try:
        with destination.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"PDF exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
                    )
                digest.update(chunk)
                handle.write(chunk)

        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        with destination.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise HTTPException(status_code=415, detail="File is not a valid PDF.")

        document = store.create_document(
            document_id=document_id,
            user_id=user_id,
            filename=file.filename or "bill.pdf",
            content_type="application/pdf",
            size_bytes=size,
            sha256=digest.hexdigest(),
            storage_path=destination,
        )
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Could not store upload: {exc}") from exc
    finally:
        await file.close()

    background_tasks.add_task(_process_document, document_id)
    return document


@router.get("", response_model=list[DocumentMeta])
def list_documents(
    user_id: Annotated[str, Depends(get_user_id)],
    limit: int = 50,
) -> list[DocumentMeta]:
    return store.list_documents(user_id, limit=max(1, min(limit, 100)))


@router.get("/{document_id}", response_model=DocumentMeta)
def get_document(
    document_id: str,
    user_id: Annotated[str, Depends(get_user_id)],
) -> DocumentMeta:
    return _require_document(document_id, user_id)


@router.get("/{document_id}/extraction", response_model=ExtractionResult)
def get_extraction(
    document_id: str,
    user_id: Annotated[str, Depends(get_user_id)],
) -> ExtractionResult:
    return _require_extraction(document_id, user_id)


@router.post("/{document_id}/confirm", response_model=ExtractionResult)
def confirm_extraction(
    document_id: str,
    payload: ConfirmRequest,
    user_id: Annotated[str, Depends(get_user_id)],
) -> ExtractionResult:
    extraction = _require_extraction(document_id, user_id)
    numeric_fields = {
        "consumption_kwh",
        "total_cost",
        "unit_rate",
        "standing_charge",
        "tax",
        "demand_kw",
    }

    def _coerce(name: str, value: Any) -> float | str | None:
        if value is None:
            return None
        if name in numeric_fields:
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=422, detail=f"{name} must be numeric."
                ) from exc
        return str(value)

    for name, value in payload.fields.items():
        value = _coerce(name, value)
        if name not in extraction.fields:
            # Allow adding a known field that the extractor missed.
            if name not in {
                "consumption_kwh",
                "total_cost",
                "unit_rate",
                "standing_charge",
                "tax",
                "demand_kw",
                "billing_period_start",
                "billing_period_end",
                "account_number",
                "meter_number",
                "tariff",
                "supplier",
                "currency",
            }:
                continue
            from .schemas import ExtractedField

            extraction.fields[name] = ExtractedField(
                name=name,
                label=name.replace("_", " ").title(),
                value=value,
                confidence=1.0,
                needs_review=False,
                notes=["Added during manual review."],
            )
            continue

        field = extraction.fields[name]
        field.value = value
        field.confidence = 1.0
        field.needs_review = False
        field.notes = [*field.notes, "Confirmed during manual review."]

    extraction.status = "confirmed"
    extraction.updated_at = utc_now()
    if payload.reviewer_note:
        extraction.warnings.append(f"Reviewer note: {payload.reviewer_note}")

    store.save_extraction(document_id, extraction)
    store.update_status(document_id, "confirmed")
    return extraction


@router.get("/{document_id}/analytics", response_model=AnalyticsResult)
def get_analytics(
    document_id: str,
    user_id: Annotated[str, Depends(get_user_id)],
) -> AnalyticsResult:
    extraction = _require_extraction(document_id, user_id)
    history = store.list_user_extractions(user_id, exclude_document_id=document_id)
    return build_analytics(extraction, history=history)


@router.get("/{document_id}/report", response_class=HTMLResponse)
def get_report(
    document_id: str,
    user_id: Annotated[str, Depends(get_user_id)],
) -> HTMLResponse:
    document = _require_document(document_id, user_id)
    extraction = _require_extraction(document_id, user_id)
    history = store.list_user_extractions(user_id, exclude_document_id=document_id)
    analytics = build_analytics(extraction, history=history)
    return HTMLResponse(render_report(document, extraction, analytics))


@router.delete("/{document_id}")
def delete_document(
    document_id: str,
    user_id: Annotated[str, Depends(get_user_id)],
) -> dict[str, Any]:
    if not store.delete_document(document_id, user_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"deleted": True, "document_id": document_id}
