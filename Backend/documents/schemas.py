"""Schemas for the LoadIQ document-intelligence MVP.

Every extracted value carries its evidence (page, line, source text and
confidence) so that charts, reports, insights and recommendations can point
back to the bill instead of making unsupported claims.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DocumentStatus = Literal[
    "queued",
    "processing",
    "needs_review",
    "confirmed",
    "failed",
]


class Evidence(BaseModel):
    id: str
    page: int = Field(ge=1)
    line: int | None = Field(default=None, ge=1)
    bbox: list[float] | None = None
    source_text: str
    method: str = "regex"
    confidence: float = Field(ge=0.0, le=1.0)
    ocr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ExtractedField(BaseModel):
    name: str
    label: str
    value: float | str | None = None
    unit: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    needs_review: bool = True
    raw_value: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    document_id: str
    status: DocumentStatus = "needs_review"
    fields: dict[str, ExtractedField] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    page_count: int = 0
    text_chars: int = 0
    extractor: str = "pypdf+regex"
    ocr_used: bool = False
    ocr_provider: str | None = None
    created_at: str
    updated_at: str


class DocumentMeta(BaseModel):
    id: str
    user_id: str
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    status: DocumentStatus
    page_count: int = 0
    text_chars: int = 0
    created_at: str
    updated_at: str
    error: str | None = None


class ConfirmRequest(BaseModel):
    fields: dict[str, Any] = Field(default_factory=dict)
    reviewer_note: str | None = None


class Insight(BaseModel):
    id: str
    title: str
    detail: str
    severity: Literal["info", "positive", "warning"] = "info"
    evidence_ids: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    id: str
    title: str
    detail: str
    estimated_savings: float | None = None
    currency: str | None = None
    confidence: Literal["low", "medium", "high"] = "medium"
    evidence_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class AnalyticsResult(BaseModel):
    document_id: str
    billing_period: dict[str, str | None] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    charts: dict[str, Any] = Field(default_factory=dict)
    insights: list[Insight] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    evidence_index: dict[str, Evidence] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    generated_at: str
