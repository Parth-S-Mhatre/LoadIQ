"""Tests for the optional nemotron-ocr-v2 fallback.

No network calls are made: the HTTP client is replaced with a fake that returns
the two response shapes documented by NVIDIA (``text_detections`` and the
legacy flat ``ocr_boxes``/``ocr_txts``/``ocr_confs`` arrays).
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

import main
from documents import extractor, ocr

BILL_LINES = [
    "ACME POWER",
    "Electricity statement",
    "Account Number: 123456789",
    "Billing Period: 01/01/2026 to 31/03/2026",
    "Total Consumption: 1,842 kWh",
    "Unit Rate: 24.5p per kWh",
    "Total Amount Due: 496.70",
]


def _blank_pdf(path: Path) -> None:
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.drawString(72, 800, " ")
    pdf.save()


def _ocr_result() -> ocr.OCRResult:
    lines = [
        ocr.OCRLine(
            text=text,
            bbox=[0.1, round(0.05 + index * 0.05, 4), 0.9, round(0.08 + index * 0.05, 4)],
            confidence=0.97,
        )
        for index, text in enumerate(BILL_LINES)
    ]
    return ocr.OCRResult(pages=[lines], provider="nemotron-ocr-v2")


def test_parse_nim_text_detections() -> None:
    payload = {
        "data": [
            {
                "text_detections": [
                    {
                        "text_prediction": {"text": "Total 1,842 kWh", "confidence": 0.93},
                        "bounding_box": {
                            "points": [[48, 60], [400, 60], [400, 80], [48, 80]]
                        },
                    }
                ]
            }
        ]
    }
    lines = ocr._parse_response(payload, width=800, height=1000)
    assert len(lines) == 1
    assert lines[0].text == "Total 1,842 kWh"
    assert lines[0].confidence == 0.93
    assert lines[0].bbox == [0.06, 0.06, 0.5, 0.08]


def test_parse_legacy_flat_arrays() -> None:
    payload = {
        "ocr_boxes": [[[15.0, 43.0], [150.0, 43.0], [150.0, 57.0], [15.0, 57.0]]],
        "ocr_txts": ["The previous notice was dated"],
        "ocr_confs": [0.9773],
    }
    lines = ocr._parse_response(payload, width=300, height=600)
    assert len(lines) == 1
    assert lines[0].text == "The previous notice was dated"
    assert lines[0].confidence == 0.9773
    assert lines[0].bbox == [0.05, 0.0717, 0.5, 0.095]


def test_group_lines_orders_words_left_to_right() -> None:
    detections = [
        ocr.OCRLine(text="Charge", bbox=[0.45, 0.2, 0.6, 0.22], confidence=0.9),
        ocr.OCRLine(text="Standing", bbox=[0.1, 0.2, 0.4, 0.22], confidence=0.95),
        ocr.OCRLine(text="Total", bbox=[0.1, 0.4, 0.2, 0.42], confidence=0.99),
    ]
    lines = ocr._group_lines(detections)
    assert [line.text for line in lines] == ["Standing Charge", "Total"]


def test_extract_pdf_uses_ocr_when_text_layer_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OCR_PROVIDER", "nemotron")
    monkeypatch.setenv("OCR_MODE", "auto")
    monkeypatch.setattr(extractor, "ocr_pdf", lambda _path: _ocr_result())

    bill = tmp_path / "scanned.pdf"
    _blank_pdf(bill)

    result = extractor.extract_pdf("doc-ocr", bill)

    assert result.ocr_used is True
    assert result.ocr_provider == "nemotron-ocr-v2"
    assert result.extractor == "pypdf+nemotron-ocr+regex"
    assert result.fields["consumption_kwh"].value == 1842.0
    assert result.fields["total_cost"].value == 496.7
    assert result.fields["unit_rate"].value == 24.5

    evidence = result.fields["consumption_kwh"].evidence[0]
    assert evidence.bbox is not None
    assert evidence.ocr_confidence == 0.97
    assert evidence.method == "regex+ocr"


def test_extract_pdf_warns_when_ocr_is_disabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OCR_PROVIDER", "none")
    monkeypatch.setenv("OCR_MODE", "auto")

    bill = tmp_path / "scanned.pdf"
    _blank_pdf(bill)

    result = extractor.extract_pdf("doc-no-ocr", bill)
    assert result.ocr_used is False
    assert any("OCR is not enabled" in warning for warning in result.warnings)


def test_ocr_health_reports_configuration(monkeypatch) -> None:
    monkeypatch.setenv("OCR_PROVIDER", "nemotron")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")

    client = TestClient(main.app)
    response = client.get("/documents/ocr/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is True
    assert payload["configured"] is True
    assert payload["api_key_configured"] is True
    assert payload["provider"] == "nemotron-ocr-v2"
    assert "endpoint" in payload
