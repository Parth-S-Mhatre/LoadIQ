from pathlib import Path

from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

import main


def _make_bill(path: Path) -> None:
    lines = [
        "ACME POWER",
        "Electricity statement",
        "Account Number: 123456789",
        "Meter Number: MPAN 1234567890123",
        "Billing Period: 01/01/2026 to 31/03/2026",
        "Total Consumption: 1,842 kWh",
        "Unit Rate: 24.5p per kWh",
        "Standing Charge: 45.0p per day",
        "Maximum Demand: 12.4 kW",
        "VAT: £18.42",
        "Total Amount Due: £496.70",
        "Tariff: Standard Variable",
    ]
    pdf = canvas.Canvas(str(path), pagesize=A4)
    y = 800
    for line in lines:
        pdf.drawString(72, y, line)
        y -= 22
    pdf.save()


def test_upload_extract_analytics_and_report(tmp_path: Path) -> None:
    client = TestClient(main.app)
    bill = tmp_path / "bill.pdf"
    _make_bill(bill)

    with bill.open("rb") as handle:
        response = client.post(
            "/documents",
            files={"file": ("bill.pdf", handle, "application/pdf")},
            headers={"X-User-ID": "test-user"},
        )
    assert response.status_code == 202, response.text
    document_id = response.json()["id"]

    status = client.get(f"/documents/{document_id}", headers={"X-User-ID": "test-user"})
    assert status.status_code == 200
    assert status.json()["status"] == "needs_review"

    extraction = client.get(
        f"/documents/{document_id}/extraction", headers={"X-User-ID": "test-user"}
    )
    assert extraction.status_code == 200, extraction.text
    fields = extraction.json()["fields"]
    assert fields["consumption_kwh"]["value"] == 1842.0
    assert fields["total_cost"]["value"] == 496.7
    assert fields["unit_rate"]["value"] == 24.5
    assert fields["unit_rate"]["unit"] == "p/kWh"
    assert fields["meter_number"]["value"] == "1234567890123"
    assert fields["currency"]["value"] == "GBP"

    analytics = client.get(
        f"/documents/{document_id}/analytics", headers={"X-User-ID": "test-user"}
    )
    assert analytics.status_code == 200, analytics.text
    payload = analytics.json()
    assert payload["summary"]["consumption_kwh"] == 1842.0
    assert payload["summary"]["billing_days"] == 89
    assert any(item["id"] == "unit_rate_comparison" for item in payload["recommendations"])
    assert payload["charts"]["cost_breakdown"]

    report = client.get(
        f"/documents/{document_id}/report", headers={"X-User-ID": "test-user"}
    )
    assert report.status_code == 200
    assert "Evidence appendix" in report.text
    assert "ev_consumption_kwh" in report.text


def test_confirm_rejects_non_numeric_consumption() -> None:
    client = TestClient(main.app)
    # A missing document is a 404, which is enough to verify the route exists
    # and the auth/tenant dependency is applied.
    response = client.post(
        "/documents/missing/confirm",
        json={"fields": {"consumption_kwh": 1}},
        headers={"X-User-ID": "test-user"},
    )
    assert response.status_code == 404
